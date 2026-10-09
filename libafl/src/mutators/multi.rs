//! Mutator definitions for [`MultipartInput`]s. See [`crate::inputs::multi`] for details.
//!
//! # Restricting mutations to certain keys with a [`MultipartMutationPolicy`]
//!
//! By default, all parts of a [`MultipartInput`] are treated equally by the built-in
//! mutators: havoc mutations pick a uniformly random part to mutate, byte-level crossover
//! exchanges bytes only between parts sharing the same key, and part-level crossover
//! exchanges arbitrary parts. A [`MultipartMutationPolicy`] in the state's metadata restricts
//! this:
//!
//! - Blacklisted keys are never chosen by the built-in havoc, crossover, and remove/replace
//!   mutators, neither as the part to mutate nor as the source or target of crossover. Parts
//!   with blacklisted keys are left entirely to special, user-defined mutators, which get
//!   full `&mut` access to the input and may freely change, drop, or duplicate them.
//!   Blacklisting wins over group membership: a key that is both blacklisted and in a group
//!   is still never crossed over.
//! - Crossover groups widen byte-level crossover from same-key to same-group: bytes and parts
//!   are only exchanged between parts whose keys share one of the configured groups, and keys
//!   in no group are not crossed over at all. At part level, a part may only replace a part
//!   with a crossover-compatible key.
//! - Byte-level crossover keeps the default behavior of appending a copy of the source part
//!   when the mutated input has no crossover-compatible part for it: groups constrain
//!   exchanges between existing parts, but they do not prevent a new (non-blacklisted) key
//!   from entering an input.
//!
//! The policy applies to the exact key type it was created with: a
//! [`MultipartMutationPolicy<String>`] only affects `MultipartInput<I, String>` inputs; a
//! policy with any other key type is never found in the metadata and stays without effect.
//!
//! Fuzzers without a [`MultipartMutationPolicy`] keep the default behavior. Special,
//! user-defined mutators may consult the policy themselves via
//! [`MultipartMutationPolicy::from_state`].
//!
//! ```rust,ignore
//! use libafl::{
//!     HasMetadata,
//!     mutators::{
//!         havoc_mutations::havoc_mutations, multi::MultipartMutationPolicy,
//!         scheduled::HavocScheduledMutator,
//!     },
//! };
//!
//! // During fuzzer setup, add the policy to the state's metadata:
//! state.add_metadata(
//!     MultipartMutationPolicy::new()
//!         // The "header" part is only ever modified by special mutators:
//!         .with_blacklist(["header".to_string()])
//!         // Crossover may only happen inside each of these key groups:
//!         .with_crossover_groups([
//!             vec!["keyA1".to_string(), "keyA2".to_string(), "keyA3".to_string()],
//!             vec!["keyB1".to_string(), "keyB2".to_string()],
//!         ]),
//! );
//!
//! // The usual havoc mutations now respect the policy:
//! let mutator = HavocScheduledMutator::new(havoc_mutations());
//! ```

use alloc::{string::String, vec::Vec};
use core::{
    cmp::{Ordering, min},
    fmt::Debug,
    num::NonZero,
};

use libafl_bolts::{Error, rands::Rand};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::HasMetadata;
use crate::{
    corpus::{Corpus, CorpusId},
    impl_default_multipart,
    inputs::{HasMutatorBytes, Input, ResizableMutator, multi::MultipartInput},
    mutators::{
        MutationResult, Mutator,
        mutations::{
            BitFlipMutator, ByteAddMutator, ByteDecMutator, ByteFlipMutator, ByteIncMutator,
            ByteInterestingMutator, ByteNegMutator, ByteRandMutator, BytesCopyMutator,
            BytesDeleteMutator, BytesExpandMutator, BytesInsertCopyMutator, BytesInsertMutator,
            BytesRandInsertMutator, BytesRandSetMutator, BytesSetMutator, BytesSwapMutator,
            CrossoverInsertMutator as BytesInputCrossoverInsertMutator,
            CrossoverReplaceMutator as BytesInputCrossoverReplaceMutator, DwordAddMutator,
            DwordInterestingMutator, QwordAddMutator, WordAddMutator, WordInterestingMutator,
            rand_range,
        },
        token_mutations::{I2SRandReplace, TokenInsert, TokenReplace},
    },
    random_corpus_id,
    state::{HasCorpus, HasMaxSize, HasRand},
};

/// A state metadata that restricts which parts of a [`MultipartInput`] the built-in mutators
/// may touch. See the [module documentation](self) for how to use it.
///
/// Without this metadata in the state, the default behavior applies: havoc mutators mutate any
/// part, byte-level crossover happens between parts with the same key, and part-level
/// crossover happens between arbitrary parts.
///
/// Blacklisting wins over crossover groups: a blacklisted key is never crossed over, even
/// when it shares a group with another key.
///
/// Special, user-defined mutators are not affected by this policy; they get full `&mut` access
/// to the input and may freely modify, drop, or duplicate any part. Use
/// [`MultipartMutationPolicy::from_state`] to consult the policy from such a mutator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipartMutationPolicy<K> {
    /// Keys that the built-in mutators never touch.
    blacklist: Vec<K>,
    /// Crossover (byte-level and part-level) only happens between keys sharing a group.
    crossover_groups: Vec<Vec<K>>,
}

impl<K> Default for MultipartMutationPolicy<K> {
    fn default() -> Self {
        Self {
            blacklist: Vec::new(),
            crossover_groups: Vec::new(),
        }
    }
}

libafl_bolts::impl_serdeany!(
    MultipartMutationPolicy<K: 'static + Debug + Serialize + DeserializeOwned>,
    <String>
);

impl<K> MultipartMutationPolicy<K>
where
    K: PartialEq,
{
    /// Creates a new [`MultipartMutationPolicy`] without any restriction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Blacklists the given keys: the built-in mutators will neither mutate parts with these
    /// keys nor exchange bytes or parts with them.
    #[must_use]
    pub fn with_blacklist(mut self, keys: impl IntoIterator<Item = K>) -> Self {
        self.blacklist.extend(keys);
        self
    }

    /// Adds crossover groups: (byte-level and part-level) crossover will only happen between
    /// parts whose keys share one of the given groups.
    ///
    /// If no group is configured, byte-level crossover only happens between parts with the
    /// same key, and part-level crossover is unrestricted.
    #[must_use]
    pub fn with_crossover_groups(
        mut self,
        groups: impl IntoIterator<Item = impl IntoIterator<Item = K>>,
    ) -> Self {
        self.crossover_groups
            .extend(groups.into_iter().map(|group| group.into_iter().collect()));
        self
    }

    /// Returns whether parts with this key may be mutated by the built-in mutators.
    #[must_use]
    pub fn is_blacklisted(&self, key: &K) -> bool {
        self.blacklist.contains(key)
    }

    /// Returns whether byte-level crossover may exchange bytes between parts with these keys.
    ///
    /// Without crossover groups, this is only allowed between equal keys (the default
    /// behavior); otherwise, both keys need to share one of the configured groups.
    #[must_use]
    pub fn byte_crossover_ok(&self, target: &K, source: &K) -> bool {
        !self.is_blacklisted(target)
            && !self.is_blacklisted(source)
            && if self.crossover_groups.is_empty() {
                target == source
            } else {
                self.same_crossover_group(target, source)
            }
    }

    /// Returns whether part-level crossover may exchange parts with these keys.
    ///
    /// Without crossover groups, any non-blacklisted pair of keys is allowed; otherwise, both
    /// keys need to share one of the configured groups.
    #[must_use]
    pub fn part_crossover_ok(&self, target: &K, source: &K) -> bool {
        !self.is_blacklisted(target)
            && !self.is_blacklisted(source)
            && (self.crossover_groups.is_empty() || self.same_crossover_group(target, source))
    }

    fn same_crossover_group(&self, first: &K, second: &K) -> bool {
        self.crossover_groups
            .iter()
            .any(|group| group.contains(first) && group.contains(second))
    }
}

impl<K> MultipartMutationPolicy<K>
where
    K: PolicyKey,
{
    /// Returns the [`MultipartMutationPolicy`] for keys of type `K` from the state's metadata,
    /// if the fuzzer added one.
    ///
    /// The built-in mutators consult the policy automatically; use this in special,
    /// user-defined mutators that want to honor the same restrictions. Note that the lookup is
    /// keyed by the exact key type `K`: a policy created for another key type is not returned.
    #[must_use]
    pub fn from_state<S>(state: &S) -> Option<&Self>
    where
        S: HasMetadata,
    {
        state.metadata_map().get::<Self>()
    }
}

/// The bounds a [`MultipartInput`] key type must satisfy to work with a
/// [`MultipartMutationPolicy`] (and therefore with the built-in policy-aware mutators).
///
/// Blanket-implemented for every type that satisfies the bounds; there is nothing to implement
/// manually. The `'static + Debug + Serialize + DeserializeOwned` part is required because the
/// policy lives in the state metadata as a [`libafl_bolts::serdeany::SerdeAny`] and is looked up
/// by its exact key type.
pub trait PolicyKey: 'static + Debug + Serialize + DeserializeOwned + PartialEq {}

impl<K> PolicyKey for K where K: 'static + Debug + Serialize + DeserializeOwned + PartialEq {}

/// Returns whether the built-in mutators may mutate the part with this key.
///
/// If the state has no [`MultipartMutationPolicy`], all keys are allowed.
#[must_use]
pub(crate) fn mutation_allowed<S, K>(state: &S, key: &K) -> bool
where
    S: HasMetadata,
    K: PolicyKey,
{
    MultipartMutationPolicy::<K>::from_state(state).is_none_or(|policy| !policy.is_blacklisted(key))
}

/// Returns whether byte-level crossover may splice bytes from the part with the `source` key
/// into the part with the `target` key.
///
/// If the state has no [`MultipartMutationPolicy`], this is only allowed for equal keys.
#[must_use]
pub(crate) fn byte_crossover_allowed<S, K>(state: &S, target: &K, source: &K) -> bool
where
    S: HasMetadata,
    K: PolicyKey,
{
    MultipartMutationPolicy::<K>::from_state(state).map_or(target == source, |policy| {
        policy.byte_crossover_ok(target, source)
    })
}

/// Returns whether part-level crossover may replace the part with the `target` key with the
/// part with the `source` key.
///
/// If the state has no [`MultipartMutationPolicy`], all pairs of keys are allowed.
#[must_use]
pub(crate) fn part_crossover_allowed<S, K>(state: &S, target: &K, source: &K) -> bool
where
    S: HasMetadata,
    K: PolicyKey,
{
    MultipartMutationPolicy::<K>::from_state(state)
        .is_none_or(|policy| policy.part_crossover_ok(target, source))
}

/// Marker trait for if the default multipart input mutator implementation is appropriate.
///
/// You should implement this type for your mutator if you just want a random part of the input to
/// be selected and mutated. Use [`impl_default_multipart`] to implement this marker trait for many
/// at once.
///
/// The selected part is the only one that will ever be mutated; a
/// [`MultipartMutationPolicy`] can restrict the choice of parts.
pub trait DefaultMultipartMutator {}

impl<I, K, M, S> Mutator<MultipartInput<I, K>, S> for M
where
    M: DefaultMultipartMutator + Mutator<I, S>,
    S: HasRand + HasMetadata,
    K: PolicyKey,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut MultipartInput<I, K>,
    ) -> Result<MutationResult, Error> {
        let mutable: Vec<usize> = input
            .parts()
            .iter()
            .enumerate()
            .filter(|&(_, (key, _))| mutation_allowed(state, key))
            .map(|(idx, _)| idx)
            .collect();
        let Some(len) = NonZero::new(mutable.len()) else {
            return Ok(MutationResult::Skipped);
        };
        let idx = mutable[state.rand_mut().below(len)];
        let (_key, part) = &mut input.parts_mut()[idx];
        self.mutate(state, part)
    }

    fn post_exec(&mut self, state: &mut S, new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        M::post_exec(self, state, new_corpus_id)
    }
}

mod macros {
    /// Implements the marker trait [`super::DefaultMultipartMutator`] for one to many types, e.g.:
    ///
    /// ```rs
    /// impl_default_multipart!(
    ///     // --- havoc ---
    ///     BitFlipMutator,
    ///     ByteAddMutator,
    ///     ByteDecMutator,
    ///     ByteFlipMutator,
    ///     ByteIncMutator,
    ///     ...
    /// );
    /// ```
    #[macro_export]
    macro_rules! impl_default_multipart {
        ($mutator: ty, $($mutators: ty),+$(,)?) => {
            impl $crate::mutators::multi::DefaultMultipartMutator for $mutator {}
            impl_default_multipart!($($mutators),+);
        };

        ($mutator: ty) => {
            impl $crate::mutators::multi::DefaultMultipartMutator for $mutator {}
        };
    }
}

impl_default_multipart!(
    // --- havoc ---
    BitFlipMutator,
    ByteAddMutator,
    ByteDecMutator,
    ByteFlipMutator,
    ByteIncMutator,
    ByteInterestingMutator,
    ByteNegMutator,
    ByteRandMutator,
    BytesCopyMutator,
    BytesDeleteMutator,
    BytesExpandMutator,
    BytesInsertCopyMutator,
    BytesInsertMutator,
    BytesRandInsertMutator,
    BytesRandSetMutator,
    BytesSetMutator,
    BytesSwapMutator,
    // crossover has a custom implementation below
    DwordAddMutator,
    DwordInterestingMutator,
    QwordAddMutator,
    WordAddMutator,
    WordInterestingMutator,
    // --- token ---
    TokenInsert,
    TokenReplace,
    // ---  i2s  ---
    I2SRandReplace,
);

impl<I, K, S> Mutator<MultipartInput<I, K>, S> for BytesInputCrossoverInsertMutator
where
    S: HasCorpus<MultipartInput<I, K>> + HasMaxSize + HasRand + HasMetadata,
    I: Input + ResizableMutator<u8> + HasMutatorBytes,
    K: Clone + PolicyKey,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut MultipartInput<I, K>,
    ) -> Result<MutationResult, Error> {
        // we can eat the slight bias; number of parts will be small
        let key_choice = state.rand_mut().next() as usize;
        let part_choice = state.rand_mut().next() as usize;

        // We special-case crossover with self
        let id = random_corpus_id!(state.corpus(), state.rand_mut());
        if let Some(cur) = state.corpus().current() {
            if id == *cur {
                // only parts that may serve as crossover source at all can be chosen
                let sources: Vec<usize> = input
                    .parts()
                    .iter()
                    .enumerate()
                    .filter(|&(_, (key, _))| byte_crossover_allowed(state, key, key))
                    .map(|(idx, _)| idx)
                    .collect();
                if sources.is_empty() {
                    return Ok(MutationResult::Skipped);
                }
                let choice = sources[key_choice % sources.len()];
                // Safety: choice is an index of an existing part
                let (key, part) = &input.parts()[choice];

                let other_size = part.mutator_bytes().len();

                if other_size < 2 {
                    return Ok(MutationResult::Skipped);
                }

                // the parts of this input that this part may crossover with
                let partners: Vec<(usize, usize)> = input
                    .parts()
                    .iter()
                    .enumerate()
                    .filter(|&(idx, (partner_key, _))| {
                        idx != choice && byte_crossover_allowed(state, partner_key, key)
                    })
                    .map(|(idx, (_, part))| (idx, part.mutator_bytes().len()))
                    .collect();

                if partners.is_empty() {
                    return Ok(MutationResult::Skipped);
                }

                let (part_idx, size) = partners[part_choice % partners.len()];
                let Some(nz) = NonZero::new(size) else {
                    return Ok(MutationResult::Skipped);
                };
                let target = state.rand_mut().below(nz);
                // # Safety
                // size is nonzero here (checked above), target is smaller than size
                // -> the subtraction result is greater than 0.
                // other_size is checked above to be larger than zero.
                let range = rand_range(state, other_size, unsafe {
                    NonZero::new(min(other_size, size - target)).unwrap_unchecked()
                });

                let [part, chosen] = match part_idx.cmp(&choice) {
                    Ordering::Less => input.parts_at_indices_mut([part_idx, choice]),
                    Ordering::Equal => {
                        unreachable!("choice should never equal the part idx!")
                    }
                    Ordering::Greater => {
                        let [chosen, part] = input.parts_at_indices_mut([choice, part_idx]);
                        [part, chosen]
                    }
                };

                return Ok(Self::crossover_insert(
                    &mut part.1,
                    size,
                    target,
                    range,
                    chosen.1.mutator_bytes(),
                ));
            }
        }

        let mut other_testcase = state.corpus().get(id)?.borrow_mut();
        let other = other_testcase.load_input(state.corpus())?;

        // only parts that may serve as crossover source at all can be chosen
        let sources: Vec<usize> = other
            .parts()
            .iter()
            .enumerate()
            .filter(|&(_, (key, _))| byte_crossover_allowed(state, key, key))
            .map(|(idx, _)| idx)
            .collect();
        if sources.is_empty() {
            return Ok(MutationResult::Skipped);
        }

        let choice = sources[key_choice % sources.len()];
        // Safety: choice is an index of an existing part
        let (key, part) = &other.parts()[choice];

        let other_size = part.mutator_bytes().len();
        if other_size < 2 {
            return Ok(MutationResult::Skipped);
        }

        // the parts of this input that bytes from this part may be inserted into
        let targets: Vec<usize> = input
            .parts()
            .iter()
            .enumerate()
            .filter(|&(_, (target_key, _))| byte_crossover_allowed(state, target_key, key))
            .map(|(idx, _)| idx)
            .collect();

        if targets.is_empty() {
            // just add it!
            input.append_part(other.part_at_index(choice).unwrap().clone());

            return Ok(MutationResult::Mutated);
        }

        let target_idx = targets[part_choice % targets.len()];
        drop(other_testcase);
        let size = input
            .part_at_index(target_idx)
            .unwrap()
            .1
            .mutator_bytes()
            .len();
        let Some(nz) = NonZero::new(size) else {
            return Ok(MutationResult::Skipped);
        };

        let target = state.rand_mut().below(nz);
        // # Safety
        // other_size is larger than 0, checked above.
        // size is larger than 0.
        // target is smaller than size -> the subtraction is larger than 0.
        let range = rand_range(state, other_size, unsafe {
            NonZero::new_unchecked(min(other_size, size - target))
        });

        let other_testcase = state.corpus().get(id)?.borrow_mut();
        // No need to load the input again, it'll still be cached.
        let other = other_testcase.input().as_ref().unwrap();

        Ok(Self::crossover_insert(
            &mut input.part_at_index_mut(target_idx).unwrap().1,
            size,
            target,
            range,
            other.part_at_index(choice).unwrap().1.mutator_bytes(),
        ))
    }
    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

impl<I, K, S> Mutator<MultipartInput<I, K>, S> for BytesInputCrossoverReplaceMutator
where
    S: HasCorpus<MultipartInput<I, K>> + HasMaxSize + HasRand + HasMetadata,
    I: Input + ResizableMutator<u8> + HasMutatorBytes,
    K: Clone + PolicyKey,
{
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut MultipartInput<I, K>,
    ) -> Result<MutationResult, Error> {
        // we can eat the slight bias; number of parts will be small
        let key_choice = state.rand_mut().next() as usize;
        let part_choice = state.rand_mut().next() as usize;

        // We special-case crossover with self
        let id = random_corpus_id!(state.corpus(), state.rand_mut());
        if let Some(cur) = state.corpus().current() {
            if id == *cur {
                // only parts that may serve as crossover source at all can be chosen
                let sources: Vec<usize> = input
                    .parts()
                    .iter()
                    .enumerate()
                    .filter(|&(_, (key, _))| byte_crossover_allowed(state, key, key))
                    .map(|(idx, _)| idx)
                    .collect();
                if sources.is_empty() {
                    return Ok(MutationResult::Skipped);
                }
                let choice = sources[key_choice % sources.len()];
                // Safety: choice is an index of an existing part
                let (key, part) = &input.parts()[choice];

                let other_size = part.mutator_bytes().len();
                if other_size < 2 {
                    return Ok(MutationResult::Skipped);
                }

                // the parts of this input that this part may crossover with
                let partners: Vec<(usize, usize)> = input
                    .parts()
                    .iter()
                    .enumerate()
                    .filter(|&(idx, (partner_key, _))| {
                        idx != choice && byte_crossover_allowed(state, partner_key, key)
                    })
                    .map(|(idx, (_, part))| (idx, part.mutator_bytes().len()))
                    .collect();

                if partners.is_empty() {
                    return Ok(MutationResult::Skipped);
                }

                let (part_idx, size) = partners[part_choice % partners.len()];

                let Some(nz) = NonZero::new(size) else {
                    return Ok(MutationResult::Skipped);
                };

                let target = state.rand_mut().below(nz);
                // # Safety
                // other_size is checked above.
                // size is larger than than target and larger than 1. The subtraction result will always be positive.
                let range = rand_range(state, other_size, unsafe {
                    NonZero::new_unchecked(min(other_size, size - target))
                });

                let [part, chosen] = match part_idx.cmp(&choice) {
                    Ordering::Less => input.parts_at_indices_mut([part_idx, choice]),
                    Ordering::Equal => {
                        unreachable!("choice should never equal the part idx!")
                    }
                    Ordering::Greater => {
                        let [chosen, part] = input.parts_at_indices_mut([choice, part_idx]);
                        [part, chosen]
                    }
                };

                return Ok(Self::crossover_replace(
                    &mut part.1,
                    target,
                    range,
                    chosen.1.mutator_bytes(),
                ));
            }
        }

        let mut other_testcase = state.corpus().get(id)?.borrow_mut();
        let other = other_testcase.load_input(state.corpus())?;

        // only parts that may serve as crossover source at all can be chosen
        let sources: Vec<usize> = other
            .parts()
            .iter()
            .enumerate()
            .filter(|&(_, (key, _))| byte_crossover_allowed(state, key, key))
            .map(|(idx, _)| idx)
            .collect();
        if sources.is_empty() {
            return Ok(MutationResult::Skipped);
        }

        let choice = sources[key_choice % sources.len()];
        // Safety: choice is an index of an existing part
        let (key, part) = &other.parts()[choice];

        let other_size = part.mutator_bytes().len();
        if other_size < 2 {
            return Ok(MutationResult::Skipped);
        }

        // the parts of this input whose bytes may be replaced by this part's bytes
        let targets: Vec<usize> = input
            .parts()
            .iter()
            .enumerate()
            .filter(|&(_, (target_key, _))| byte_crossover_allowed(state, target_key, key))
            .map(|(idx, _)| idx)
            .collect();

        if targets.is_empty() {
            // just add it!
            input.append_part(other.part_at_index(choice).unwrap().clone());

            return Ok(MutationResult::Mutated);
        }

        let target_idx = targets[part_choice % targets.len()];
        drop(other_testcase);
        let size = input
            .part_at_index(target_idx)
            .unwrap()
            .1
            .mutator_bytes()
            .len();
        let Some(nz) = NonZero::new(size) else {
            return Ok(MutationResult::Skipped);
        };

        let target = state.rand_mut().below(nz);
        // # Safety
        // other_size is checked above.
        // size is larger than than target and larger than 1. The subtraction result will always be positive.
        let range = rand_range(state, other_size, unsafe {
            NonZero::new_unchecked(min(other_size, size - target))
        });

        let other_testcase = state.corpus().get(id)?.borrow_mut();
        // No need to load the input again, it'll still be cached.
        let other = other_testcase.input().as_ref().unwrap();

        Ok(Self::crossover_replace(
            &mut input.part_at_index_mut(target_idx).unwrap().1,
            target,
            range,
            other.part_at_index(choice).unwrap().1.mutator_bytes(),
        ))
    }
    #[inline]
    fn post_exec(&mut self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{borrow::Cow, string::String, string::ToString};

    use libafl_bolts::{Named, rands::StdRand};

    use super::*;
    use crate::{
        corpus::InMemoryCorpus,
        feedbacks::ConstFeedback,
        inputs::BytesInput,
        mutators::list::{
            CrossoverInsertMutator as PartCrossoverInsertMutator,
            CrossoverReplaceMutator as PartCrossoverReplaceMutator, RemoveLastEntryMutator,
            RemoveRandomEntryMutator,
        },
        state::StdState,
    };

    type TestInput = MultipartInput<BytesInput, String>;

    fn input(parts: &[(&str, &[u8])]) -> TestInput {
        TestInput::new(
            parts
                .iter()
                .map(|&(key, bytes)| (key.to_string(), BytesInput::new(bytes.to_vec())))
                .collect(),
        )
    }

    fn corpus_with(inputs: &[TestInput]) -> InMemoryCorpus<TestInput> {
        let mut corpus = InMemoryCorpus::new();
        for input in inputs {
            corpus.add(input.clone().into()).unwrap();
        }
        corpus
    }

    fn test_state(
        corpus: InMemoryCorpus<TestInput>,
        policy: Option<MultipartMutationPolicy<String>>,
    ) -> impl HasCorpus<TestInput> + HasMetadata + HasRand + HasMaxSize {
        let mut feedback = ConstFeedback::new(false);
        let mut objective = ConstFeedback::new(false);

        let mut state = StdState::new(
            StdRand::with_seed(0),
            corpus,
            InMemoryCorpus::new(),
            &mut feedback,
            &mut objective,
        )
        .unwrap();
        if let Some(policy) = policy {
            state.add_metadata(policy);
        }
        state
    }

    /// A mutator that appends a single byte, to observe which part was picked.
    struct AppendByteMutator;

    impl<S> Mutator<BytesInput, S> for AppendByteMutator {
        fn mutate(
            &mut self,
            _state: &mut S,
            input: &mut BytesInput,
        ) -> Result<MutationResult, Error> {
            input.resize(input.mutator_bytes().len() + 1, b'X');
            Ok(MutationResult::Mutated)
        }

        fn post_exec(
            &mut self,
            _state: &mut S,
            _new_corpus_id: Option<CorpusId>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    impl Named for AppendByteMutator {
        fn name(&self) -> &Cow<'static, str> {
            static NAME: Cow<'static, str> = Cow::Borrowed("AppendByteMutator");
            &NAME
        }
    }

    impl DefaultMultipartMutator for AppendByteMutator {}

    #[test]
    fn policy_queries() {
        let policy = MultipartMutationPolicy::new()
            .with_blacklist(["bl".to_string()])
            .with_crossover_groups([
                vec!["a1".to_string(), "a2".to_string()],
                vec!["b1".to_string()],
            ]);
        assert!(policy.is_blacklisted(&"bl".to_string()));
        assert!(!policy.is_blacklisted(&"a1".to_string()));

        // byte crossover: same group if groups are configured
        assert!(policy.byte_crossover_ok(&"a1".to_string(), &"a2".to_string()));
        assert!(!policy.byte_crossover_ok(&"a1".to_string(), &"b1".to_string()));
        assert!(!policy.byte_crossover_ok(&"a1".to_string(), &"bl".to_string()));
        assert!(!policy.byte_crossover_ok(&"ungrouped".to_string(), &"ungrouped".to_string()));

        // part crossover: same group if groups are configured
        assert!(policy.part_crossover_ok(&"a1".to_string(), &"a2".to_string()));
        assert!(!policy.part_crossover_ok(&"a1".to_string(), &"b1".to_string()));
        assert!(!policy.part_crossover_ok(&"bl".to_string(), &"bl".to_string()));
        assert!(!policy.part_crossover_ok(&"a1".to_string(), &"ungrouped".to_string()));

        // an empty policy keeps the default behavior
        let empty = MultipartMutationPolicy::<String>::new();
        assert!(empty.byte_crossover_ok(&"x".to_string(), &"x".to_string()));
        assert!(!empty.byte_crossover_ok(&"x".to_string(), &"y".to_string()));
        assert!(empty.part_crossover_ok(&"x".to_string(), &"y".to_string()));
    }

    #[test]
    fn havoc_mutates_any_part_without_policy() {
        let mut state = test_state(corpus_with(&[]), None);
        let mut inp = input(&[("a", b""), ("b", b"")]);
        let mut mutator = AppendByteMutator;
        for _ in 0..100 {
            assert_eq!(
                mutator.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Mutated
            );
        }
        assert_eq!(inp.len(), 2);
        assert_eq!(
            inp.parts()[0].1.mutator_bytes().len() + inp.parts()[1].1.mutator_bytes().len(),
            100
        );
    }

    #[test]
    fn havoc_skips_blacklisted_keys() {
        let policy = MultipartMutationPolicy::new().with_blacklist(["b".to_string()]);
        let mut state = test_state(corpus_with(&[]), Some(policy));
        let mut inp = input(&[("a", b"x"), ("b", b"y")]);
        let mut mutator = AppendByteMutator;
        for _ in 0..100 {
            mutator.mutate(&mut state, &mut inp).unwrap();
        }
        // only "a" was ever picked
        assert_eq!(inp.parts()[0].1.mutator_bytes().len(), 101);
        assert_eq!(inp.parts()[1].1.mutator_bytes(), "y".as_bytes());

        // nothing to mutate if all keys are blacklisted
        let policy =
            MultipartMutationPolicy::new().with_blacklist(["a".to_string(), "b".to_string()]);
        let mut state = test_state(corpus_with(&[]), Some(policy));
        let mut inp = input(&[("a", b"x"), ("b", b"y")]);
        for _ in 0..10 {
            assert_eq!(
                mutator.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Skipped
            );
        }
        assert_eq!(inp.parts()[0].1.mutator_bytes(), "x".as_bytes());
        assert_eq!(inp.parts()[1].1.mutator_bytes(), "y".as_bytes());
    }

    #[test]
    fn byte_crossover_defaults_to_same_key() {
        let other = input(&[("a", &[0xaa; 32]), ("b", &[0xbb; 32])]);
        let mut state = test_state(corpus_with(&[other]), None);

        let mut insert = BytesInputCrossoverInsertMutator::new();
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            insert.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xaa)
        );

        let mut replace = BytesInputCrossoverReplaceMutator::new();
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xaa)
        );
    }

    #[test]
    fn byte_crossover_skips_blacklisted_keys() {
        let policy = MultipartMutationPolicy::new().with_blacklist(["b".to_string()]);
        let other = input(&[("a", &[0xaa; 32]), ("b", &[0xbb; 32])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));

        let mut insert = BytesInputCrossoverInsertMutator::new();
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            insert.mutate(&mut state, &mut inp).unwrap();
        }
        // the blacklisted part is never source, target, or appended
        assert_eq!(inp.len(), 2);
        assert_eq!(inp.parts()[1].1.mutator_bytes(), [0x22; 8].as_slice());

        let mut replace = BytesInputCrossoverReplaceMutator::new();
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert_eq!(inp.parts()[1].1.mutator_bytes(), [0x22; 8].as_slice());
    }

    #[test]
    fn byte_crossover_stays_inside_groups() {
        let policy = MultipartMutationPolicy::new().with_crossover_groups([
            vec!["a1".to_string(), "a2".to_string()],
            vec!["b1".to_string(), "b2".to_string()],
        ]);
        let other = input(&[("a2", &[0xaa; 32]), ("b2", &[0xbb; 32])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));

        let mut insert = BytesInputCrossoverInsertMutator::new();
        let mut inp = input(&[("a1", &[0x11; 8]), ("b1", &[0x22; 8]), ("c", &[0x33; 8])]);
        for _ in 0..100 {
            insert.mutate(&mut state, &mut inp).unwrap();
        }
        // ungrouped "c" is never chosen, and the groups never mix
        assert_eq!(inp.len(), 3);
        assert_eq!(inp.parts()[2].1.mutator_bytes(), [0x33; 8].as_slice());
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xaa)
        );

        let mut replace = BytesInputCrossoverReplaceMutator::new();
        let mut inp = input(&[("a1", &[0x11; 8]), ("b1", &[0x22; 8]), ("c", &[0x33; 8])]);
        for _ in 0..100 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 3);
        assert_eq!(inp.parts()[2].1.mutator_bytes(), [0x33; 8].as_slice());
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xaa)
        );
    }

    #[test]
    fn byte_crossover_with_self_stays_inside_groups() {
        let policy = MultipartMutationPolicy::new()
            .with_crossover_groups([vec!["a1".to_string(), "a2".to_string()]]);
        let mut state = test_state(corpus_with(&[]), Some(policy));
        // crossover will pick the only corpus entry, which is also the current one
        let id = state
            .corpus_mut()
            .add(input(&[("x", &[0x55; 32])]).into())
            .unwrap();
        *state.corpus_mut().current_mut() = Some(id);

        let mut insert = BytesInputCrossoverInsertMutator::new();
        let mut inp = input(&[("a1", &[0x11; 8]), ("a2", &[0x22; 8]), ("b1", &[0x33; 8])]);
        for _ in 0..100 {
            insert.mutate(&mut state, &mut inp).unwrap();
        }
        // ungrouped "b1" is never chosen for crossover with self
        assert_eq!(inp.len(), 3);
        assert_eq!(inp.parts()[2].1.mutator_bytes(), [0x33; 8].as_slice());
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte == 0x11 || byte == 0x22)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte == 0x11 || byte == 0x22)
        );
    }

    #[test]
    fn part_mutators_skip_blacklisted_keys() {
        let policy = MultipartMutationPolicy::new().with_blacklist(["keep".to_string()]);
        let mut state = test_state(corpus_with(&[]), Some(policy.clone()));

        // remove never deletes the blacklisted part
        let mut remove = RemoveRandomEntryMutator;
        let mut inp = input(&[("a", b"x"), ("keep", b"y"), ("b", b"z")]);
        for _ in 0..100 {
            remove.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 1);
        assert_eq!(inp.parts()[0].0, "keep");
        assert_eq!(inp.parts()[0].1.mutator_bytes(), "y".as_bytes());

        // remove-last skips instead of deleting a blacklisted part
        let mut remove_last = RemoveLastEntryMutator;
        let mut inp = input(&[("a", b"x"), ("keep", b"y")]);
        assert_eq!(
            remove_last.mutate(&mut state, &mut inp).unwrap(),
            MutationResult::Skipped
        );
        assert_eq!(inp.len(), 2);

        // part crossover does not insert blacklisted parts
        let other = input(&[("keep", &[0x77; 8])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy.clone()));
        let mut insert = PartCrossoverInsertMutator;
        let mut inp = input(&[("a", b"x")]);
        for _ in 0..50 {
            assert_eq!(
                insert.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Skipped
            );
        }
        assert_eq!(inp.len(), 1);

        // part crossover never replaces a blacklisted target
        let other = input(&[("keep", &[0x77; 8]), ("src", &[0x88; 8])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));
        let mut replace = PartCrossoverReplaceMutator;
        let mut inp = input(&[("keep", b"y"), ("a", b"x")]);
        for _ in 0..50 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert_eq!(inp.parts()[0].0, "keep");
        assert_eq!(inp.parts()[0].1.mutator_bytes(), "y".as_bytes());
    }

    #[test]
    fn part_crossover_replace_stays_inside_groups() {
        let policy = MultipartMutationPolicy::new().with_crossover_groups([
            vec!["a1".to_string(), "a2".to_string()],
            vec!["b1".to_string(), "b2".to_string()],
        ]);
        let other = input(&[("a2", &[0xaa; 16]), ("b2", &[0xbb; 16])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));

        let mut replace = PartCrossoverReplaceMutator;
        let mut inp = input(&[("a1", &[0x11; 8]), ("b1", &[0x22; 8])]);
        for _ in 0..100 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        // parts are only ever replaced by parts from the same group
        assert_eq!(inp.len(), 2);
        assert!(inp.parts()[0].0 == "a1" || inp.parts()[0].0 == "a2");
        assert!(inp.parts()[1].0 == "b1" || inp.parts()[1].0 == "b2");
    }

     #[test]
    fn empty_policy_matches_default_behavior() {
        let policy = MultipartMutationPolicy::<String>::default();
        let other = input(&[("a", &[0xaa; 32]), ("b", &[0xbb; 32])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));

        // byte crossover still only exchanges between the same keys and never appends here
        let mut insert = BytesInputCrossoverInsertMutator;
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            assert_eq!(
                insert.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Mutated
            );
        }
        assert_eq!(inp.len(), 2);
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );
        assert!(
            inp.parts()[1]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xaa)
        );

        // part crossover may still replace any part with any other
        let mut replace = PartCrossoverReplaceMutator;
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            assert_eq!(
                replace.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Mutated
            );
        }
        assert_eq!(inp.len(), 2);

        // remove still reaches every part, down to the empty input
        let mut remove = RemoveRandomEntryMutator;
        let mut inp = input(&[("a", b"x"), ("b", b"y")]);
        for _ in 0..2 {
            assert_eq!(
                remove.mutate(&mut state, &mut inp).unwrap(),
                MutationResult::Mutated
            );
        }
        assert_eq!(inp.len(), 0);
        assert_eq!(
            remove.mutate(&mut state, &mut inp).unwrap(),
            MutationResult::Skipped
        );
    }

    #[test]
    fn blacklist_wins_over_crossover_group() {
        let policy = MultipartMutationPolicy::new()
            .with_blacklist(["b".to_string()])
            .with_crossover_groups([vec!["a".to_string(), "b".to_string()]]);
        let other = input(&[("a", &[0xaa; 32]), ("b", &[0xbb; 32])]);
        let mut state = test_state(corpus_with(&[other]), Some(policy));

        // byte crossover never uses the blacklisted part, even as same-group partner
        let mut insert = BytesInputCrossoverInsertMutator;
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            insert.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert_eq!(inp.parts()[1].1.mutator_bytes(), [0x22; 8].as_slice());
        assert!(
            inp.parts()[0]
                .1
                .mutator_bytes()
                .iter()
                .all(|&byte| byte != 0xbb)
        );

        // part crossover: blacklisted target untouched, blacklisted source never picked
        let mut replace = PartCrossoverReplaceMutator;
        let mut inp = input(&[("a", &[0x11; 8]), ("b", &[0x22; 8])]);
        for _ in 0..100 {
            replace.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.len(), 2);
        assert_eq!(inp.parts()[0].0, "a");
        assert_eq!(inp.parts()[1].0, "b");
        assert_eq!(inp.parts()[1].1.mutator_bytes(), [0x22; 8].as_slice());

        // havoc skips the blacklisted part, even inside a group
        let mut append = AppendByteMutator;
        let mut inp = input(&[("a", b"x"), ("b", b"y")]);
        for _ in 0..20 {
            append.mutate(&mut state, &mut inp).unwrap();
        }
        assert_eq!(inp.parts()[1].1.mutator_bytes(), "y".as_bytes());
    }

    #[test]
    fn from_state_finds_only_the_matching_key_type() {
        let policy = MultipartMutationPolicy::new().with_blacklist(["b".to_string()]);
        let state = test_state(corpus_with(&[]), Some(policy));
        let found = MultipartMutationPolicy::<String>::from_state(&state).unwrap();
        assert!(found.is_blacklisted(&"b".to_string()));
        // a policy created for another key type is never found
        assert!(MultipartMutationPolicy::<u32>::from_state(&state).is_none());

        let state = test_state(corpus_with(&[]), None);
        assert!(MultipartMutationPolicy::<String>::from_state(&state).is_none());
    }
}
