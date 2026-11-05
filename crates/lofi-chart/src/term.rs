//! Native storage for many-sorted, locally nameless terms.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::thread::{self, ThreadId};

use num_bigint::BigUint;

/// A dense sort identifier within one signature.
pub type SortId = u16;

/// A dense operator identifier within one signature.
pub type OperatorId = u32;

/// A dense free-variable identifier within one lineage.
pub type AtomId = u32;

/// A node identifier, unique within one lineage generation.
pub type Serial = u64;

/// The number of serial bits reserved for a region's local node index.
pub const GENERATION_SHIFT: u32 = 40;

const LOCAL_INDEX_LIMIT: usize = 1_usize << GENERATION_SHIFT;
const FRESH_PREFIX: char = '\0';

static NEXT_LINEAGE: AtomicU64 = AtomicU64::new(1);

/// The non-term payload accepted by an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PayloadKind {
    /// The operator accepts no payload.
    None,
    /// The payload is a stable vocabulary identifier.
    Constant,
    /// The payload is a de Bruijn index.
    Index,
    /// The payload is a free-variable handle.
    Free,
}

/// One variable introduced by a binding operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingSite {
    /// Argument positions under the site.
    pub arguments: Vec<usize>,
    /// The variable's sort.
    pub sort: SortId,
}

/// Static facts about one operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSpec {
    /// Stable structural name.
    pub name: String,
    /// Result sort.
    pub sort: SortId,
    /// Required argument sorts.
    pub arguments: Vec<SortId>,
    /// Payload convention.
    pub payload: PayloadKind,
    /// Feature-erased operator.
    pub erased: OperatorId,
    /// Binding sites in declaration order.
    pub binding: Vec<BindingSite>,
    /// Per-argument, per-sort binder shifts.
    pub shifts: Vec<Vec<(SortId, u32)>>,
}

impl OperatorSpec {
    fn shift(&self, position: usize, sort: SortId) -> u32 {
        self.shifts[position]
            .iter()
            .find_map(|(held, amount)| (*held == sort).then_some(*amount))
            .unwrap_or(0)
    }
}

/// A validated term signature lowered once at the language boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureSpec {
    /// Signature name.
    pub name: String,
    /// Sort names by dense identifier.
    pub sorts: Vec<String>,
    /// Operators by dense identifier.
    pub operators: Vec<OperatorSpec>,
    /// Vocabulary sizes by sort.
    pub vocabularies: Vec<Option<usize>>,
    /// Direct feature-core identifier for each vocabulary constant.
    pub constant_cores: Vec<Option<Vec<usize>>>,
    /// Constant operators by sort.
    pub constants: Vec<Option<OperatorId>>,
    /// Bound-variable operators by sort.
    pub bound: Vec<Option<OperatorId>>,
    /// Free-variable operators by sort.
    pub free: Vec<Option<OperatorId>>,
    /// Hole operators by sort.
    pub holes: Vec<Option<OperatorId>>,
    /// Whether binders may be vacuous.
    pub allows_vacuous_binding: bool,
}

/// The stable identity part of a node payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Identity {
    /// No payload.
    None,
    /// A constant identifier or bound index.
    Integer(u64),
    /// A free variable's stable name.
    Name(String),
}

/// A store-independent structural key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TermKey {
    /// Operator name.
    pub operator: String,
    /// Stable payload identity.
    pub identity: Identity,
    /// Argument keys.
    pub arguments: Vec<Self>,
}

#[derive(Debug)]
struct Node {
    operator: OperatorId,
    operator_name: String,
    sort: SortId,
    arguments: Vec<Term>,
    payload: Option<u64>,
    identity: Identity,
    core: Option<Term>,
    size: usize,
    unbound: Vec<(SortId, u32)>,
    free: BigUint,
    serial: Serial,
    lineage: u64,
    generation: u32,
    structural_hash: u64,
}

/// An immutable native term node.
#[derive(Clone, Debug)]
pub struct Term(Arc<Node>);

impl Term {
    /// Return the root operator identifier.
    #[must_use]
    pub fn operator(&self) -> OperatorId {
        self.0.operator
    }

    /// Return the root operator's stable name.
    #[must_use]
    pub fn operator_name(&self) -> &str {
        &self.0.operator_name
    }

    /// Return the result sort identifier.
    #[must_use]
    pub fn sort(&self) -> SortId {
        self.0.sort
    }

    /// Return the arguments in declaration order.
    #[must_use]
    pub fn arguments(&self) -> &[Self] {
        &self.0.arguments
    }

    /// Return the local payload.
    #[must_use]
    pub fn payload(&self) -> Option<u64> {
        self.0.payload
    }

    /// Return the store-independent payload identity.
    #[must_use]
    pub fn identity(&self) -> &Identity {
        &self.0.identity
    }

    /// Return the feature-erased term.
    #[must_use]
    pub fn core(&self) -> Self {
        self.0.core.clone().unwrap_or_else(|| self.clone())
    }

    /// Return the node count, including repeated positions.
    #[must_use]
    pub fn size(&self) -> usize {
        self.0.size
    }

    /// Return one past the greatest open index of any sort.
    #[must_use]
    pub fn max_index(&self) -> u32 {
        self.0
            .unbound
            .iter()
            .map(|(_, bound)| *bound)
            .max()
            .unwrap_or(0)
    }

    /// Return one past the greatest open index of one sort.
    #[must_use]
    pub fn max_index_of(&self, sort: SortId) -> u32 {
        self.0
            .unbound
            .iter()
            .find_map(|(held, bound)| (*held == sort).then_some(*bound))
            .unwrap_or(0)
    }

    /// Return open-index bounds in sort order.
    #[must_use]
    pub fn unbound(&self) -> &[(SortId, u32)] {
        &self.0.unbound
    }

    /// Return the free-variable mask.
    #[must_use]
    pub fn free(&self) -> &BigUint {
        &self.0.free
    }

    /// Return the packed generation and local index.
    #[must_use]
    pub fn serial(&self) -> Serial {
        self.0.serial
    }

    /// Return the publishing generation.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.0.generation
    }

    /// Return whether no bound index is open.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.0.unbound.is_empty()
    }

    /// Return a complete store-independent structural key.
    #[must_use]
    pub fn key(&self) -> TermKey {
        let root = Arc::as_ptr(&self.0);
        let mut built: HashMap<*const Node, TermKey> = HashMap::new();
        let mut pending = vec![(self.clone(), false)];

        while let Some((term, expanded)) = pending.pop() {
            let pointer = Arc::as_ptr(&term.0);
            if built.contains_key(&pointer) {
                continue;
            }
            if expanded {
                let arguments = term
                    .arguments()
                    .iter()
                    .map(|argument| built[&Arc::as_ptr(&argument.0)].clone())
                    .collect();
                let key = TermKey {
                    operator: term.operator_name().to_owned(),
                    identity: term.identity().clone(),
                    arguments,
                };
                if pointer == root {
                    return key;
                }
                built.insert(pointer, key);
            } else {
                pending.push((term.clone(), true));
                pending.extend(
                    term.arguments()
                        .iter()
                        .rev()
                        .filter(|argument| !built.contains_key(&Arc::as_ptr(&argument.0)))
                        .cloned()
                        .map(|argument| (argument, false)),
                );
            }
        }
        TermKey {
            operator: self.operator_name().to_owned(),
            identity: self.identity().clone(),
            arguments: Vec::new(),
        }
    }

    /// Return the fixed structural hash.
    #[must_use]
    pub fn structural_hash(&self) -> u64 {
        self.0.structural_hash
    }

    /// Return whether two handles address the same interned node.
    #[must_use]
    pub fn is_identical(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Return whether the term was published in the lineage and generation set.
    #[must_use]
    pub fn is_from(&self, lineage: u64, ancestry: &[u32]) -> bool {
        self.0.lineage == lineage && ancestry.contains(&self.0.generation)
    }

    fn structurally_equal(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        if self.0.structural_hash != other.0.structural_hash {
            return false;
        }
        let mut pending = vec![(self.clone(), other.clone())];
        while let Some((left, right)) = pending.pop() {
            if Arc::ptr_eq(&left.0, &right.0) {
                continue;
            }
            if left.operator_name() != right.operator_name()
                || left.identity() != right.identity()
                || left.arguments().len() != right.arguments().len()
            {
                return false;
            }
            pending.extend(
                left.arguments()
                    .iter()
                    .cloned()
                    .zip(right.arguments().iter().cloned()),
            );
        }
        true
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        self.structurally_equal(other)
    }
}

impl Eq for Term {}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.structural_hash.hash(state);
    }
}

/// A refusal at the native term boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermError {
    /// An operator identifier is outside the signature.
    UnknownOperator {
        /// Supplied identifier.
        operator: OperatorId,
    },
    /// An argument count differs from the declaration.
    Arity {
        /// Applied operator.
        operator: String,
        /// Declared arity.
        expected: usize,
        /// Supplied argument count.
        received: usize,
    },
    /// An argument has the wrong sort.
    Sort {
        /// Applied operator.
        operator: String,
        /// Argument position.
        position: usize,
        /// Declared sort.
        expected: SortId,
        /// Supplied sort.
        received: SortId,
    },
    /// A payload violates its operator's convention.
    Payload {
        /// Applied operator.
        operator: String,
        /// Supplied payload.
        payload: Option<u64>,
        /// Refusal detail.
        reason: String,
    },
    /// An argument is outside the store lineage.
    Foreign {
        /// Applied operator.
        operator: String,
        /// Argument position.
        position: usize,
        /// Receiving generation.
        expected: u32,
        /// Argument generation.
        received: u32,
    },
    /// A local region has exhausted its node serials.
    Full {
        /// Region name.
        region: String,
        /// Local node cap.
        capacity: usize,
        /// Requested node count.
        size: usize,
    },
    /// An operation is invalid in a region's current state.
    Region {
        /// Region name.
        region: String,
        /// Refused operation.
        operation: String,
        /// Region state.
        state: String,
    },
    /// A classifier declaration conflicts with an existing declaration.
    Classification {
        /// Refusal detail.
        reason: String,
    },
    /// A free-variable handle is absent or belongs to another sort.
    Atom {
        /// Supplied atom handle.
        index: AtomId,
        /// Required sort, when one was required.
        sort: Option<SortId>,
    },
    /// A requested derived operator is absent.
    MissingDerived {
        /// Derived operator kind.
        kind: &'static str,
        /// Requested sort.
        sort: SortId,
    },
    /// A replacement with open indices would be captured.
    NotLocallyClosed {
        /// Required enclosing depth.
        max_index: u32,
        /// Available depth.
        depth: u32,
    },
    /// A binder receives the wrong number of atoms.
    BindingAtoms {
        /// Binding operator.
        operator: String,
        /// Binding-site count.
        expected: usize,
        /// Supplied atom count.
        received: usize,
    },
    /// A strict signature rejects a vacuous binder.
    VacuousBinding {
        /// Binding operator.
        operator: String,
        /// Unused variable name.
        variable: String,
    },
}

impl fmt::Display for TermError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TermError {}

#[derive(Debug, Default)]
struct Atoms {
    ids: HashMap<(SortId, String), AtomId>,
    values: Vec<(SortId, String)>,
    by_sort: HashMap<SortId, Vec<AtomId>>,
}

#[derive(Debug)]
struct Lineage {
    id: u64,
    atoms: Mutex<Atoms>,
    next_generation: Mutex<u32>,
}

#[derive(Debug)]
struct Region {
    name: String,
    generation: u32,
    max_nodes: usize,
    active: bool,
    active_children: usize,
    owner: Option<ThreadId>,
    ancestry: Vec<u32>,
    parent: Option<Weak<StoreInner>>,
    table: HashMap<u64, Vec<Term>>,
    recent: Option<Term>,
    count: usize,
    hits: usize,
    misses: usize,
}

#[derive(Debug)]
struct StoreInner {
    signature: Arc<SignatureSpec>,
    lineage: Arc<Lineage>,
    region: Mutex<Region>,
    classifiers: Arc<Mutex<HashMap<SortId, Vec<Option<Term>>>>>,
}

struct PreparedNode {
    lineage: u64,
    core: Option<Term>,
    size: usize,
    unbound: Vec<(SortId, u32)>,
    free: BigUint,
}

/// A thread-safe, region-aware native term store.
#[derive(Clone, Debug)]
pub struct TermStore(Arc<StoreInner>);

impl TermStore {
    /// Construct a root store for a validated signature.
    ///
    /// # Errors
    ///
    /// Returns an error when the local node cap is invalid.
    pub fn new(signature: SignatureSpec, max_nodes: usize) -> Result<Self, TermError> {
        if max_nodes == 0 || max_nodes > LOCAL_INDEX_LIMIT {
            return Err(TermError::Full {
                region: "root".to_owned(),
                capacity: LOCAL_INDEX_LIMIT,
                size: max_nodes,
            });
        }
        if signature.constant_cores.len() != signature.sorts.len() {
            return Err(TermError::Classification {
                reason: format!(
                    "constant-core rows number {}, but the signature has {} sorts",
                    signature.constant_cores.len(),
                    signature.sorts.len()
                ),
            });
        }
        for (sort, (vocabulary, cores)) in signature
            .vocabularies
            .iter()
            .zip(&signature.constant_cores)
            .enumerate()
        {
            match (vocabulary, cores) {
                (None, None) => {}
                (Some(size), Some(row)) if row.len() == *size => {
                    for (constant, core) in row.iter().copied().enumerate() {
                        if core >= *size || row[core] != core {
                            return Err(TermError::Classification {
                                reason: format!(
                                    "constant core {core} for identifier {constant} of sort {sort} is not a direct identifier in 0..{size}"
                                ),
                            });
                        }
                    }
                }
                (None, Some(_)) => {
                    return Err(TermError::Classification {
                        reason: format!("sort {sort} has constant cores but no vocabulary"),
                    });
                }
                (Some(size), Some(row)) => {
                    return Err(TermError::Classification {
                        reason: format!(
                            "constant-core row for sort {sort} has {} entries, not {size}",
                            row.len()
                        ),
                    });
                }
                (Some(_), None) => {
                    return Err(TermError::Classification {
                        reason: format!("vocabulary sort {sort} has no constant-core row"),
                    });
                }
            }
        }
        let lineage = Arc::new(Lineage {
            id: NEXT_LINEAGE.fetch_add(1, Ordering::Relaxed),
            atoms: Mutex::new(Atoms::default()),
            next_generation: Mutex::new(1),
        });
        Ok(Self(Arc::new(StoreInner {
            signature: Arc::new(signature),
            lineage,
            region: Mutex::new(Region {
                name: "root".to_owned(),
                generation: 0,
                max_nodes,
                active: true,
                active_children: 0,
                owner: None,
                ancestry: vec![0],
                parent: None,
                table: HashMap::new(),
                recent: None,
                count: 0,
                hits: 0,
                misses: 0,
            }),
            classifiers: Arc::new(Mutex::new(HashMap::new())),
        })))
    }

    fn locked_region(&self) -> MutexGuard<'_, Region> {
        self.0
            .region
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn require_usable(region: &Region, operation: &str) -> Result<(), TermError> {
        if !region.active {
            return Err(TermError::Region {
                region: region.name.clone(),
                operation: operation.to_owned(),
                state: "it is released".to_owned(),
            });
        }
        if region
            .owner
            .is_some_and(|owner| owner != thread::current().id())
        {
            return Err(TermError::Region {
                region: region.name.clone(),
                operation: operation.to_owned(),
                state: "it belongs to another thread".to_owned(),
            });
        }
        Ok(())
    }

    /// Return the signature.
    #[must_use]
    pub fn signature(&self) -> &SignatureSpec {
        &self.0.signature
    }

    /// Return the region name.
    #[must_use]
    pub fn name(&self) -> String {
        self.locked_region().name.clone()
    }

    /// Return the local node cap.
    #[must_use]
    pub fn max_nodes(&self) -> usize {
        self.locked_region().max_nodes
    }

    /// Return the local node count.
    #[must_use]
    pub fn local_nodes(&self) -> usize {
        self.locked_region().count
    }

    /// Return whether the region is active.
    #[must_use]
    pub fn active(&self) -> bool {
        self.locked_region().active
    }

    /// Return the region generation.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.locked_region().generation
    }

    /// Return the interning hit count.
    #[must_use]
    pub fn hits(&self) -> usize {
        self.locked_region().hits
    }

    /// Return the interning miss count.
    #[must_use]
    pub fn misses(&self) -> usize {
        self.locked_region().misses
    }

    fn operator(&self, operator: OperatorId) -> Result<&OperatorSpec, TermError> {
        self.0
            .signature
            .operators
            .get(usize::try_from(operator).unwrap_or(usize::MAX))
            .ok_or(TermError::UnknownOperator { operator })
    }

    fn identity(&self, op: &OperatorSpec, payload: Option<u64>) -> Result<Identity, TermError> {
        match op.payload {
            PayloadKind::None if payload.is_none() => Ok(Identity::None),
            PayloadKind::None => Err(TermError::Payload {
                operator: op.name.clone(),
                payload,
                reason: "it takes no payload".to_owned(),
            }),
            PayloadKind::Constant => {
                let size = self.0.signature.vocabularies[usize::from(op.sort)].unwrap_or(0);
                match payload {
                    Some(value) if usize::try_from(value).is_ok_and(|held| held < size) => {
                        Ok(Identity::Integer(value))
                    }
                    _ => Err(TermError::Payload {
                        operator: op.name.clone(),
                        payload,
                        reason: format!(
                            "its vocabulary holds the identifiers 0 to {}",
                            size.saturating_sub(1)
                        ),
                    }),
                }
            }
            PayloadKind::Index => {
                payload
                    .map(Identity::Integer)
                    .ok_or_else(|| TermError::Payload {
                        operator: op.name.clone(),
                        payload,
                        reason: "it takes an integer payload".to_owned(),
                    })
            }
            PayloadKind::Free => {
                let index = payload
                    .and_then(|value| AtomId::try_from(value).ok())
                    .ok_or_else(|| TermError::Payload {
                        operator: op.name.clone(),
                        payload,
                        reason: "it takes a free-variable handle".to_owned(),
                    })?;
                let atoms = self
                    .0
                    .lineage
                    .atoms
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match atoms
                    .values
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                {
                    Some((sort, name)) if *sort == op.sort => Ok(Identity::Name(name.clone())),
                    _ => Err(TermError::Payload {
                        operator: op.name.clone(),
                        payload,
                        reason: format!(
                            "it is not a handle on a free variable of sort {:?}",
                            self.0.signature.sorts[usize::from(op.sort)]
                        ),
                    }),
                }
            }
        }
    }

    fn structural_hash(op: &OperatorSpec, identity: &Identity, arguments: &[Term]) -> u64 {
        let mut hasher = DefaultHasher::new();
        op.name.hash(&mut hasher);
        identity.hash(&mut hasher);
        for argument in arguments {
            argument.0.structural_hash.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn lookup_chain(
        &self,
        hash: u64,
        op: &OperatorSpec,
        identity: &Identity,
        arguments: &[Term],
    ) -> Option<Term> {
        let mut cursor = Some(self.0.clone());
        while let Some(inner) = cursor {
            let region = inner
                .region
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(found) = region.table.get(&hash).and_then(|bucket| {
                bucket.iter().find(|term| {
                    term.operator_name() == op.name
                        && term.identity() == identity
                        && term.arguments().len() == arguments.len()
                        && term
                            .arguments()
                            .iter()
                            .zip(arguments)
                            .all(|(left, right)| left == right)
                })
            }) {
                let found = found.clone();
                drop(region);
                return Some(found);
            }
            cursor = region.parent.as_ref().and_then(Weak::upgrade);
        }
        None
    }

    fn prepare_node(
        &self,
        operator: OperatorId,
        op: &OperatorSpec,
        arguments: &[Term],
        payload: Option<u64>,
    ) -> Result<PreparedNode, TermError> {
        let (lineage, generation, ancestry) = {
            let region = self.locked_region();
            (
                self.0.lineage.id,
                region.generation,
                region.ancestry.clone(),
            )
        };
        for (position, argument) in arguments.iter().enumerate() {
            if !argument.is_from(lineage, &ancestry) {
                return Err(TermError::Foreign {
                    operator: op.name.clone(),
                    position,
                    expected: generation,
                    received: argument.generation(),
                });
            }
        }

        let cores: Vec<_> = arguments.iter().map(Term::core).collect();
        let core_payload = match (op.payload, payload) {
            (PayloadKind::Constant, Some(identifier)) => self.0.signature.constant_cores
                [usize::from(op.sort)]
            .as_ref()
            .and_then(|row| usize::try_from(identifier).ok().and_then(|at| row.get(at)))
            .and_then(|core| u64::try_from(*core).ok()),
            _ => payload,
        };
        let core = if op.erased == operator
            && core_payload == payload
            && cores
                .iter()
                .zip(arguments)
                .all(|(left, right)| Arc::ptr_eq(&left.0, &right.0))
        {
            None
        } else {
            Some(self.intern(op.erased, cores, core_payload)?)
        };
        let mut size = 1;
        let mut free = BigUint::from(0_u8);
        let mut unbound: BTreeMap<SortId, u32> = BTreeMap::new();
        for (position, argument) in arguments.iter().enumerate() {
            size += argument.size();
            free |= argument.free();
            for (sort, bound) in argument.unbound() {
                let left = bound.saturating_sub(op.shift(position, *sort));
                if left > *unbound.get(sort).unwrap_or(&0) {
                    unbound.insert(*sort, left);
                }
            }
        }
        if let Some(value) = payload {
            match op.payload {
                PayloadKind::Index => {
                    let bound = u32::try_from(value.saturating_add(1)).unwrap_or(u32::MAX);
                    unbound
                        .entry(op.sort)
                        .and_modify(|held| *held = (*held).max(bound))
                        .or_insert(bound);
                }
                PayloadKind::Free => {
                    free |= BigUint::from(1_u8) << usize::try_from(value).unwrap_or(usize::MAX);
                }
                PayloadKind::None | PayloadKind::Constant => {}
            }
        }
        unbound.retain(|_, bound| *bound != 0);
        Ok(PreparedNode {
            lineage,
            core,
            size,
            unbound: unbound.into_iter().collect(),
            free,
        })
    }

    /// Intern one operator application.
    ///
    /// # Errors
    ///
    /// Returns a typed refusal for an invalid application, payload, lineage or full region.
    pub fn intern(
        &self,
        operator: OperatorId,
        arguments: Vec<Term>,
        payload: Option<u64>,
    ) -> Result<Term, TermError> {
        {
            let region = self.locked_region();
            Self::require_usable(&region, "intern terms")?;
        }
        let op = self.operator(operator)?.clone();
        if arguments.len() != op.arguments.len() {
            return Err(TermError::Arity {
                operator: op.name,
                expected: op.arguments.len(),
                received: arguments.len(),
            });
        }
        for (position, (argument, expected)) in arguments.iter().zip(&op.arguments).enumerate() {
            if argument.sort() != *expected {
                return Err(TermError::Sort {
                    operator: op.name.clone(),
                    position,
                    expected: *expected,
                    received: argument.sort(),
                });
            }
        }
        let identity = self.identity(&op, payload)?;
        let hash = Self::structural_hash(&op, &identity, &arguments);
        if let Some(found) = self.lookup_chain(hash, &op, &identity, &arguments) {
            let mut region = self.locked_region();
            region.hits += 1;
            region.recent = Some(found.clone());
            drop(region);
            return Ok(found);
        }
        let prepared = self.prepare_node(operator, &op, &arguments, payload)?;
        let mut region = self.locked_region();
        Self::require_usable(&region, "intern terms")?;
        if let Some(found) = region.table.get(&hash).and_then(|bucket| {
            bucket.iter().find(|term| {
                term.operator_name() == op.name
                    && term.identity() == &identity
                    && term.arguments() == arguments
            })
        }) {
            let found = found.clone();
            region.hits += 1;
            region.recent = Some(found.clone());
            drop(region);
            return Ok(found);
        }
        if region.count >= region.max_nodes {
            return Err(TermError::Full {
                region: region.name.clone(),
                capacity: region.max_nodes,
                size: region.count + 1,
            });
        }
        let serial = (u64::from(region.generation) << GENERATION_SHIFT)
            | u64::try_from(region.count).unwrap_or(u64::MAX);
        let node = Term(Arc::new(Node {
            operator,
            operator_name: op.name,
            sort: op.sort,
            arguments,
            payload,
            identity,
            core: prepared.core,
            size: prepared.size,
            unbound: prepared.unbound,
            free: prepared.free,
            serial,
            lineage: prepared.lineage,
            generation: region.generation,
            structural_hash: hash,
        }));
        region.table.entry(hash).or_default().push(node.clone());
        region.recent = Some(node.clone());
        region.count += 1;
        region.misses += 1;
        drop(region);
        Ok(node)
    }

    /// Return the named constant.
    ///
    /// # Errors
    ///
    /// Returns an error when the sort has no constant operator or the identifier is invalid.
    pub fn constant(&self, sort: SortId, identifier: u64) -> Result<Term, TermError> {
        let operator = self
            .0
            .signature
            .constants
            .get(usize::from(sort))
            .copied()
            .flatten()
            .ok_or(TermError::MissingDerived {
                kind: "constant",
                sort,
            })?;
        self.intern(operator, vec![], Some(identifier))
    }

    /// Return a bound variable.
    ///
    /// # Errors
    ///
    /// Returns an error when the sort has no bound-variable operator.
    pub fn bound(&self, sort: SortId, index: u64) -> Result<Term, TermError> {
        let operator = self
            .0
            .signature
            .bound
            .get(usize::from(sort))
            .copied()
            .flatten()
            .ok_or(TermError::MissingDerived {
                kind: "bound variable",
                sort,
            })?;
        self.intern(operator, vec![], Some(index))
    }

    /// Return a free variable.
    ///
    /// # Errors
    ///
    /// Returns an error when the sort is not open or the store is unavailable.
    pub fn free(&self, sort: SortId, name: &str) -> Result<Term, TermError> {
        let operator = self
            .0
            .signature
            .free
            .get(usize::from(sort))
            .copied()
            .flatten()
            .ok_or(TermError::MissingDerived {
                kind: "free variable",
                sort,
            })?;
        let atom = self.atom(sort, name)?;
        self.intern(operator, vec![], Some(u64::from(atom)))
    }

    /// Return the hole of one sort.
    ///
    /// # Errors
    ///
    /// Returns an error when the signature declares no hole for the sort.
    pub fn hole(&self, sort: SortId) -> Result<Term, TermError> {
        let operator = self
            .0
            .signature
            .holes
            .get(usize::from(sort))
            .copied()
            .flatten()
            .ok_or(TermError::MissingDerived { kind: "hole", sort })?;
        self.intern(operator, vec![], None)
    }

    /// Intern a free-variable name.
    ///
    /// # Errors
    ///
    /// Returns an error when the sort is not open or the store is unavailable.
    pub fn atom(&self, sort: SortId, name: &str) -> Result<AtomId, TermError> {
        {
            let region = self.locked_region();
            Self::require_usable(&region, "intern atoms")?;
        }
        if self
            .0
            .signature
            .free
            .get(usize::from(sort))
            .copied()
            .flatten()
            .is_none()
        {
            return Err(TermError::MissingDerived {
                kind: "free variable",
                sort,
            });
        }
        let mut atoms = self
            .0
            .lineage
            .atoms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = (sort, name.to_owned());
        if let Some(index) = atoms.ids.get(&key) {
            let index = *index;
            drop(atoms);
            return Ok(index);
        }
        let index = AtomId::try_from(atoms.values.len()).map_err(|_| TermError::Atom {
            index: AtomId::MAX,
            sort: Some(sort),
        })?;
        atoms.values.push(key.clone());
        atoms.by_sort.entry(sort).or_default().push(index);
        atoms.ids.insert(key, index);
        drop(atoms);
        Ok(index)
    }

    /// Resolve a free-variable handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the handle is absent or the store is unavailable.
    pub fn atom_value(&self, index: AtomId) -> Result<(SortId, String), TermError> {
        let region = self.locked_region();
        Self::require_usable(&region, "look up atoms")?;
        drop(region);
        self.0
            .lineage
            .atoms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .cloned()
            .ok_or(TermError::Atom { index, sort: None })
    }

    /// Return fresh atom handles outside a mask.
    ///
    /// # Errors
    ///
    /// Returns an error when the sort is not open or the store is unavailable.
    pub fn fresh(
        &self,
        sort: SortId,
        avoid: &BigUint,
        count: usize,
    ) -> Result<Vec<AtomId>, TermError> {
        let mut chosen = Vec::with_capacity(count);
        let known = self
            .0
            .lineage
            .atoms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .by_sort
            .get(&sort)
            .cloned()
            .unwrap_or_default();
        for index in known {
            if chosen.len() == count {
                break;
            }
            if ((avoid >> usize::try_from(index).unwrap_or(usize::MAX)) & BigUint::from(1_u8))
                == BigUint::from(0_u8)
            {
                chosen.push(index);
            }
        }
        let mut minted = 0;
        while chosen.len() < count {
            let index = self.atom(sort, &format!("{FRESH_PREFIX}{minted}"))?;
            minted += 1;
            if ((avoid >> usize::try_from(index).unwrap_or(usize::MAX)) & BigUint::from(1_u8))
                == BigUint::from(0_u8)
                && !chosen.contains(&index)
            {
                chosen.push(index);
            }
        }
        Ok(chosen)
    }

    /// Return whether a term belongs to this active lineage.
    ///
    /// # Errors
    ///
    /// Returns an error when the store is unavailable.
    pub fn contains(&self, term: &Term) -> Result<bool, TermError> {
        let region = self.locked_region();
        Self::require_usable(&region, "inspect term ownership")?;
        Ok(term.is_from(self.0.lineage.id, &region.ancestry))
    }

    /// Store one immutable classifier vector on the root store.
    ///
    /// # Errors
    ///
    /// Returns an error for a child store, foreign term, conflict or unavailable store.
    pub fn classify(&self, sort: SortId, classifiers: Vec<Option<Term>>) -> Result<(), TermError> {
        let region = self.locked_region();
        Self::require_usable(&region, "classify constants")?;
        if region.parent.is_some() {
            return Err(TermError::Region {
                region: region.name.clone(),
                operation: "classify constants".to_owned(),
                state: "it is a child region".to_owned(),
            });
        }
        let ancestry = region.ancestry.clone();
        let generation = region.generation;
        drop(region);
        for term in classifiers.iter().flatten() {
            if !term.is_from(self.0.lineage.id, &ancestry) {
                return Err(TermError::Foreign {
                    operator: "#classifier".to_owned(),
                    position: 0,
                    expected: generation,
                    received: term.generation(),
                });
            }
        }
        let mut held = self
            .0
            .classifiers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(known) = held.get(&sort) {
            if known != &classifiers {
                drop(held);
                return Err(TermError::Classification {
                    reason: "the constants are already classified".to_owned(),
                });
            }
            drop(held);
            return Ok(());
        }
        held.insert(sort, classifiers);
        drop(held);
        Ok(())
    }

    /// Return the declared classifier at a vocabulary identifier.
    #[must_use]
    pub fn classifier(&self, sort: SortId, identifier: usize) -> Option<Term> {
        self.0
            .classifiers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&sort)
            .and_then(|classifiers| classifiers.get(identifier))
            .cloned()
            .flatten()
    }

    /// Open a child region.
    ///
    /// # Errors
    ///
    /// Returns an error when the cap is invalid or the parent store is unavailable.
    pub fn child(&self, name: String, max_nodes: usize) -> Result<Self, TermError> {
        if max_nodes == 0 || max_nodes > LOCAL_INDEX_LIMIT {
            return Err(TermError::Full {
                region: name,
                capacity: LOCAL_INDEX_LIMIT,
                size: max_nodes,
            });
        }
        let ancestry = {
            let mut parent = self.locked_region();
            Self::require_usable(&parent, "open child regions")?;
            parent.active_children += 1;
            parent.ancestry.clone()
        };
        let generation = {
            let mut next = self
                .0
                .lineage
                .next_generation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let held = *next;
            *next += 1;
            held
        };
        let mut child_ancestry = ancestry;
        child_ancestry.push(generation);
        Ok(Self(Arc::new(StoreInner {
            signature: self.0.signature.clone(),
            lineage: self.0.lineage.clone(),
            region: Mutex::new(Region {
                name,
                generation,
                max_nodes,
                active: true,
                active_children: 0,
                owner: Some(thread::current().id()),
                ancestry: child_ancestry,
                parent: Some(Arc::downgrade(&self.0)),
                table: HashMap::new(),
                recent: None,
                count: 0,
                hits: 0,
                misses: 0,
            }),
            classifiers: self.0.classifiers.clone(),
        })))
    }

    /// Release a child region. Repeated releases are inert.
    ///
    /// # Errors
    ///
    /// Returns an error for a root store, active child or wrong owning thread.
    pub fn release(&self) -> Result<(), TermError> {
        let parent = {
            let mut region = self.locked_region();
            if region.parent.is_none() {
                return Err(TermError::Region {
                    region: region.name.clone(),
                    operation: "release".to_owned(),
                    state: "it is the root store".to_owned(),
                });
            }
            if !region.active {
                return Ok(());
            }
            Self::require_usable(&region, "release")?;
            if region.active_children != 0 {
                return Err(TermError::Region {
                    region: region.name.clone(),
                    operation: "release".to_owned(),
                    state: "it has active child regions".to_owned(),
                });
            }
            let parent = region.parent.as_ref().and_then(Weak::upgrade);
            region.active = false;
            region.table.clear();
            region.recent = None;
            region.count = 0;
            parent
        };
        if let Some(parent) = parent {
            let mut region = parent
                .region
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            region.active_children -= 1;
        }
        Ok(())
    }

    /// Promote a term into the immediate parent region.
    ///
    /// # Errors
    ///
    /// Returns an error for a root store, foreign term or unavailable region.
    pub fn promote(&self, term: &Term) -> Result<Term, TermError> {
        let parent = {
            let region = self.locked_region();
            Self::require_usable(&region, "promote terms")?;
            region
                .parent
                .as_ref()
                .and_then(Weak::upgrade)
                .ok_or_else(|| TermError::Region {
                    region: region.name.clone(),
                    operation: "promote terms".to_owned(),
                    state: "it is the root store".to_owned(),
                })?
        };
        let parent = Self(parent);
        let mut memo = HashMap::new();
        self.promote_into(term, &parent, &mut memo)
    }

    fn promote_into(
        &self,
        term: &Term,
        parent: &Self,
        memo: &mut HashMap<Serial, Term>,
    ) -> Result<Term, TermError> {
        if parent.contains(term)? {
            return Ok(term.clone());
        }
        if !self.contains(term)? {
            let region = self.locked_region();
            return Err(TermError::Region {
                region: region.name.clone(),
                operation: "promote terms".to_owned(),
                state: "the term is outside its lineage".to_owned(),
            });
        }
        if let Some(held) = memo.get(&term.serial()) {
            return Ok(held.clone());
        }
        let arguments = term
            .arguments()
            .iter()
            .map(|argument| self.promote_into(argument, parent, memo))
            .collect::<Result<Vec<_>, _>>()?;
        let promoted = parent.intern(term.operator(), arguments, term.payload())?;
        memo.insert(term.serial(), promoted.clone());
        Ok(promoted)
    }

    fn rebuild(&self, term: &Term, arguments: Vec<Term>) -> Result<Term, TermError> {
        if term
            .arguments()
            .iter()
            .zip(&arguments)
            .all(|(left, right)| Arc::ptr_eq(&left.0, &right.0))
        {
            Ok(term.clone())
        } else {
            self.intern(term.operator(), arguments, term.payload())
        }
    }

    /// Replace one closed bound variable occurrence.
    ///
    /// # Errors
    ///
    /// Returns an error when the replacement is open or rebuilding is refused.
    pub fn instantiate(&self, body: &Term, value: &Term, depth: u32) -> Result<Term, TermError> {
        if !value.is_closed() {
            return Err(TermError::NotLocallyClosed {
                max_index: value.max_index(),
                depth,
            });
        }
        let sort = value.sort();
        self.transform(
            body,
            depth,
            &mut |store, term, at| {
                if term.max_index_of(sort) <= at {
                    return Ok(Some(term.clone()));
                }
                let op = store.operator(term.operator())?;
                if op.payload == PayloadKind::Index
                    && term.sort() == sort
                    && term.payload() == Some(u64::from(at))
                {
                    return Ok(Some(value.clone()));
                }
                Ok(None)
            },
            sort,
        )
    }

    /// Replace one free variable with a bound index.
    ///
    /// # Errors
    ///
    /// Returns an error when the atom is absent or rebuilding is refused.
    pub fn abstract_atom(&self, body: &Term, atom: AtomId, depth: u32) -> Result<Term, TermError> {
        let (sort, _) = self.atom_value(atom)?;
        let bit = BigUint::from(1_u8) << usize::try_from(atom).unwrap_or(usize::MAX);
        let bound = self.0.signature.bound[usize::from(sort)].ok_or(TermError::MissingDerived {
            kind: "bound variable",
            sort,
        })?;
        self.transform(
            body,
            depth,
            &mut |store, term, at| {
                if (term.free() & &bit) == BigUint::from(0_u8) {
                    return Ok(Some(term.clone()));
                }
                let op = store.operator(term.operator())?;
                if op.payload == PayloadKind::Free && term.payload() == Some(u64::from(atom)) {
                    return Ok(Some(store.intern(bound, vec![], Some(u64::from(at)))?));
                }
                Ok(None)
            },
            sort,
        )
    }

    fn transform<F>(
        &self,
        term: &Term,
        at: u32,
        replace: &mut F,
        sort: SortId,
    ) -> Result<Term, TermError>
    where
        F: FnMut(&Self, &Term, u32) -> Result<Option<Term>, TermError>,
    {
        if let Some(replacement) = replace(self, term, at)? {
            return Ok(replacement);
        }
        let op = self.operator(term.operator())?.clone();
        let arguments = term
            .arguments()
            .iter()
            .enumerate()
            .map(|(position, argument)| {
                self.transform(argument, at + op.shift(position, sort), replace, sort)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.rebuild(term, arguments)
    }

    /// Replace one free variable without changing bound indices.
    ///
    /// # Errors
    ///
    /// Returns an error when the atom is absent, the value is open or rebuilding is refused.
    pub fn substitute(&self, body: &Term, atom: AtomId, value: &Term) -> Result<Term, TermError> {
        let bit = BigUint::from(1_u8) << usize::try_from(atom).unwrap_or(usize::MAX);
        if (body.free() & &bit) == BigUint::from(0_u8) {
            return Ok(body.clone());
        }
        let op = self.operator(body.operator())?.clone();
        if op.payload == PayloadKind::Free && body.payload() == Some(u64::from(atom)) {
            return Ok(value.clone());
        }
        let arguments = body
            .arguments()
            .iter()
            .map(|argument| self.substitute(argument, atom, value))
            .collect::<Result<Vec<_>, _>>()?;
        self.rebuild(body, arguments)
    }

    /// Bind one named variable under a single-site operator.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid binder, missing atom, vacuity or refused node.
    pub fn bind(
        &self,
        operator: OperatorId,
        variable: &str,
        arguments: Vec<Term>,
    ) -> Result<Term, TermError> {
        let op = self.operator(operator)?.clone();
        if op.binding.len() != 1 {
            return Err(TermError::BindingAtoms {
                operator: op.name,
                expected: op.binding.len(),
                received: 1,
            });
        }
        let site = &op.binding[0];
        let atom = self.atom(site.sort, variable)?;
        let bit = BigUint::from(1_u8) << usize::try_from(atom).unwrap_or(usize::MAX);
        if !self.0.signature.allows_vacuous_binding
            && !site.arguments.iter().any(|position| {
                arguments
                    .get(*position)
                    .is_some_and(|argument| (argument.free() & &bit) != BigUint::from(0_u8))
            })
        {
            return Err(TermError::VacuousBinding {
                operator: op.name,
                variable: variable.to_owned(),
            });
        }
        let held = arguments
            .into_iter()
            .enumerate()
            .map(|(position, argument)| {
                if site.arguments.contains(&position) {
                    self.abstract_atom(&argument, atom, 0)
                } else {
                    Ok(argument)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.intern(operator, held, None)
    }

    fn sites_at(
        op: &OperatorSpec,
        position: usize,
        atoms: &[AtomId],
    ) -> BTreeMap<SortId, Vec<AtomId>> {
        let mut grouped: BTreeMap<SortId, Vec<AtomId>> = BTreeMap::new();
        for (site, atom) in op.binding.iter().zip(atoms) {
            if site.arguments.contains(&position) {
                grouped.entry(site.sort).or_default().push(*atom);
            }
        }
        grouped
    }

    fn open_term(
        &self,
        term: &Term,
        replacements: &BTreeMap<SortId, Vec<AtomId>>,
        depth: &BTreeMap<SortId, u32>,
    ) -> Result<Term, TermError> {
        if replacements
            .iter()
            .all(|(sort, _)| term.max_index_of(*sort) <= depth.get(sort).copied().unwrap_or(0))
        {
            return Ok(term.clone());
        }
        let op = self.operator(term.operator())?.clone();
        if op.payload == PayloadKind::Index
            && let Some(replacements) = replacements.get(&op.sort)
        {
            let held = term.payload().unwrap_or(0);
            let offset = u64::from(depth.get(&op.sort).copied().unwrap_or(0));
            if let Some(position) = held
                .checked_sub(offset)
                .and_then(|value| usize::try_from(value).ok())
                && let Some(atom) = replacements.get(position)
            {
                let free = self.0.signature.free[usize::from(op.sort)].ok_or(
                    TermError::MissingDerived {
                        kind: "free variable",
                        sort: op.sort,
                    },
                )?;
                return self.intern(free, vec![], Some(u64::from(*atom)));
            }
        }
        let arguments = term
            .arguments()
            .iter()
            .enumerate()
            .map(|(position, argument)| {
                let below = replacements
                    .keys()
                    .map(|sort| {
                        (
                            *sort,
                            depth.get(sort).copied().unwrap_or(0) + op.shift(position, *sort),
                        )
                    })
                    .collect();
                self.open_term(argument, replacements, &below)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.rebuild(term, arguments)
    }

    /// Open every site of a binding term with fresh free variables.
    ///
    /// # Errors
    ///
    /// Returns an error when the binder is invalid or rebuilding is refused.
    pub fn open_binder(&self, binder: &Term) -> Result<(Term, Vec<AtomId>), TermError> {
        let op = self.operator(binder.operator())?.clone();
        let mut taken = binder.free().clone();
        let mut atoms = Vec::with_capacity(op.binding.len());
        for site in &op.binding {
            let atom = self.fresh(site.sort, &taken, 1)?[0];
            taken |= BigUint::from(1_u8) << usize::try_from(atom).unwrap_or(usize::MAX);
            atoms.push(atom);
        }
        let arguments = binder
            .arguments()
            .iter()
            .enumerate()
            .map(|(position, argument)| {
                let replacements = Self::sites_at(&op, position, &atoms);
                let depth = replacements.keys().map(|sort| (*sort, 0)).collect();
                self.open_term(argument, &replacements, &depth)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((self.rebuild(binder, arguments)?, atoms))
    }

    /// Close free variables at every site of a binding term.
    ///
    /// # Errors
    ///
    /// Returns an error when the atom count differs or rebuilding is refused.
    pub fn close_binder(&self, binder: &Term, atoms: &[AtomId]) -> Result<Term, TermError> {
        let op = self.operator(binder.operator())?.clone();
        if atoms.len() != op.binding.len() {
            return Err(TermError::BindingAtoms {
                operator: op.name,
                expected: op.binding.len(),
                received: atoms.len(),
            });
        }
        let arguments = binder
            .arguments()
            .iter()
            .enumerate()
            .map(|(position, argument)| {
                let mut held = argument.clone();
                for variables in Self::sites_at(&op, position, atoms).values() {
                    for (depth, atom) in variables.iter().enumerate() {
                        held = self.abstract_atom(
                            &held,
                            *atom,
                            u32::try_from(depth).unwrap_or(u32::MAX),
                        )?;
                    }
                }
                Ok(held)
            })
            .collect::<Result<Vec<_>, TermError>>()?;
        self.rebuild(binder, arguments)
    }
}

#[cfg(test)]
mod tests {
    use super::{OperatorSpec, PayloadKind, SignatureSpec, TermStore};

    fn signature() -> SignatureSpec {
        SignatureSpec {
            name: "tiny".to_owned(),
            sorts: vec!["T".to_owned()],
            operators: vec![
                OperatorSpec {
                    name: "leaf".to_owned(),
                    sort: 0,
                    arguments: vec![],
                    payload: PayloadKind::None,
                    erased: 0,
                    binding: vec![],
                    shifts: vec![],
                },
                OperatorSpec {
                    name: "pair".to_owned(),
                    sort: 0,
                    arguments: vec![0, 0],
                    payload: PayloadKind::None,
                    erased: 1,
                    binding: vec![],
                    shifts: vec![vec![], vec![]],
                },
            ],
            vocabularies: vec![None],
            constant_cores: vec![None],
            constants: vec![None],
            bound: vec![None],
            free: vec![None],
            holes: vec![None],
            allows_vacuous_binding: true,
        }
    }

    fn constant_signature(cores: Vec<usize>) -> SignatureSpec {
        SignatureSpec {
            name: "constants".to_owned(),
            sorts: vec!["Cat".to_owned()],
            operators: vec![OperatorSpec {
                name: "#const[Cat]".to_owned(),
                sort: 0,
                arguments: vec![],
                payload: PayloadKind::Constant,
                erased: 0,
                binding: vec![],
                shifts: vec![],
            }],
            vocabularies: vec![Some(3)],
            constant_cores: vec![Some(cores)],
            constants: vec![Some(0)],
            bound: vec![None],
            free: vec![None],
            holes: vec![None],
            allows_vacuous_binding: true,
        }
    }

    #[test]
    fn interns_one_structure_once() -> Result<(), super::TermError> {
        let store = TermStore::new(signature(), 16)?;
        let leaf = store.intern(0, vec![], None)?;
        let pair = store.intern(1, vec![leaf.clone(), leaf.clone()], None)?;
        let again = store.intern(1, vec![leaf.clone(), leaf], None)?;
        assert_eq!(pair.serial(), again.serial());
        assert_eq!(pair.size(), 3);
        assert_eq!(store.local_nodes(), 2);
        assert_eq!(store.hits(), 1);
        Ok(())
    }

    #[test]
    fn regions_share_ancestors_and_release_local_storage() -> Result<(), super::TermError> {
        let root = TermStore::new(signature(), 16)?;
        let leaf = root.intern(0, vec![], None)?;
        let child = root.child("sentence".to_owned(), 4)?;
        assert_eq!(child.intern(0, vec![], None)?, leaf);
        let pair = child.intern(1, vec![leaf.clone(), leaf], None)?;
        assert_eq!(pair.generation(), 1);
        child.release()?;
        assert!(!child.active());
        Ok(())
    }

    #[test]
    fn vocabulary_constants_use_their_declared_direct_core() -> Result<(), super::TermError> {
        let store = TermStore::new(constant_signature(vec![0, 0, 0]), 16)?;
        let base = store.constant(0, 0)?;
        let featured = store.constant(0, 1)?;

        assert!(base.core().is_identical(&base));
        assert!(featured.core().is_identical(&base));
        assert!(featured.core().core().is_identical(&base));
        Ok(())
    }

    #[test]
    fn vocabulary_constant_cores_must_be_direct() {
        let result = TermStore::new(constant_signature(vec![0, 2, 0]), 16);
        assert!(matches!(
            result,
            Err(super::TermError::Classification { .. })
        ));
    }
}
