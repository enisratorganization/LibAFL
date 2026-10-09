//! An input composed of multiple parts identified by a key.

use alloc::{string::String, vec::Vec};
use core::{fmt::Debug, hash::Hash};

#[cfg(feature = "std")]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use core::any::{Any, TypeId};
#[cfg(feature = "std")]
use std::{fs::File, io::Read, path::Path};

use serde::{Serialize, de::DeserializeOwned};

#[cfg(feature = "std")]
use crate::inputs::{BytesInput, HasMutatorBytes};
use crate::{
    corpus::CorpusId,
    inputs::{Input, ListInput},
};
#[cfg(feature = "std")]
use libafl_bolts::{Error, fs::write_file_atomic};

/// An input composed of multiple parts, each identified by a key.
///
/// It relies on a list to store the keys and parts. Keys may appear multiple times.
///
/// [`Input::to_file`]/[`Input::from_file`] use a JSON on-disk format
/// (`[{"key":"..","value":"<hex>"}]`, see the method docs) for the
/// `MultipartInput<BytesInput, String>` instantiation; every other instantiation uses the
/// generic `postcard` format of the [`Input`] trait defaults.
pub type MultipartInput<I, K> = ListInput<(K, I)>;

impl<I, K> Input for MultipartInput<I, K>
where
    I: Input + 'static,
    K: PartialEq + Debug + Serialize + DeserializeOwned + Clone + Hash + 'static,
{
    fn generate_name(&self, id: Option<CorpusId>) -> String {
        self.parts()
            .iter()
            .map(|(_k, i)| i.generate_name(id))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Write this input to a file.
    ///
    /// For `MultipartInput<BytesInput, String>` (checked at runtime via [`TypeId`]), this
    /// writes the JSON format described in [`MultipartInput`], where each part is a
    /// `{"key":"..","value":"<hex>"}` object (lowercase hex, order preserved).
    /// All other instantiations use the `postcard` format of the [`Input::to_file`] default.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or serialized.
    #[cfg(feature = "std")]
    fn to_file<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        if TypeId::of::<(K, I)>() == TypeId::of::<(String, BytesInput)>() {
            // The TypeId check above proves both Vec types are the same, so the downcast cannot fail.
            let boxed: Box<dyn Any> = Box::new(self.parts().to_vec());
            let parts = boxed
                .downcast_ref::<Vec<(String, BytesInput)>>()
                .ok_or_else(|| Error::illegal_state("impossible multipart key/value type mixup"))?;
            let json_parts: Vec<JsonPart> = parts
                .iter()
                .map(|(key, value)| JsonPart {
                    key: key.clone(),
                    value: hex_encode(value.mutator_bytes()),
                })
                .collect();
            let json = serde_json::to_vec(&json_parts).map_err(|err| {
                Error::serialize(format!(
                    "failed to serialize multipart input as JSON: {err}"
                ))
            })?;
            write_file_atomic(path, &json)
        } else {
            // duplicate of the Input::to_file default (it cannot be called from an override)
            write_file_atomic(path, &postcard::to_allocvec(self)?)
        }
    }

    /// Load an input from a file written by [`MultipartInput::to_file`].
    ///
    /// For `MultipartInput<BytesInput, String>` (checked at runtime via [`TypeId`]), the file
    /// is parsed as the JSON format described in [`MultipartInput`]: an array (order
    /// preserved) of `{"key":"..","value":"<hex>"}` objects, keys plain strings, values
    /// hex-encoded bytes (lower- or uppercase). Duplicate keys are allowed.
    /// All other instantiations use the `postcard` format of the [`Input::from_file`] default.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, does not match the format above, or a
    /// `value` is not valid hex.
    #[cfg(feature = "std")]
    fn from_file<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        if TypeId::of::<(K, I)>() == TypeId::of::<(String, BytesInput)>() {
            let content = std::fs::read_to_string(path.as_ref())?;
            let json_parts: Vec<JsonPart> = serde_json::from_str(&content).map_err(|err| {
                Error::serialize(format!(
                    "{} is not valid multipart input JSON: {err}",
                    path.as_ref().display()
                ))
            })?;
            let parts: Vec<(String, BytesInput)> = json_parts
                .into_iter()
                .map(|part| Ok((part.key, BytesInput::new(hex_value_to_bytes(&part.value)?))))
                .collect::<Result<Vec<_>, Error>>()?;
            // The TypeId check above proves both Vec types are the same, so the downcast cannot fail.
            let boxed: Box<dyn Any> = Box::new(parts);
            Ok(Self::new(*boxed.downcast::<Vec<(K, I)>>().map_err(
                |_| Error::illegal_state("impossible multipart key/value type mixup"),
            )?))
        } else {
            // duplicate of the Input::from_file default (it cannot be called from an override)
            let mut file = File::open(path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            Ok(postcard::from_bytes(&bytes)?)
        }
    }
}

/// One `{"key": .., "value": ".."}` entry of the JSON on-disk format of
/// [`MultipartInput<BytesInput, String>`].
#[cfg(feature = "std")]
#[derive(Serialize, serde::Deserialize)]
struct JsonPart {
    key: String,
    value: String,
}

/// Encodes bytes as a lowercase hex string, the inverse of [`hex_value_to_bytes`].
#[cfg(feature = "std")]
fn hex_encode(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// Decodes a lowercase/uppercase hex string (even length, empty allowed).
#[cfg(feature = "std")]
fn hex_value_to_bytes(hex: &str) -> Result<Vec<u8>, Error> {
    let bytes = hex.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(Error::illegal_argument(format!(
            "hex value {hex:?} does not have an even number of digits"
        )));
    }
    bytes
        .chunks_exact(2)
        .map(|pair| {
            let nibble = |c: u8| match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                b'A'..=b'F' => Ok(c - b'A' + 10),
                _ => Err(Error::illegal_argument(format!(
                    "invalid hex character '{}' in {hex:?}",
                    c as char
                ))),
            };
            Ok((nibble(pair[0])? << 4) | nibble(pair[1])?)
        })
        .collect()
}

/// Trait for types that provide a way to access parts by key.
pub trait Keyed<K, V> {
    /// Get the keys of the parts of this input.
    ///
    /// Keys may appear multiple times if they are used multiple times in the input.
    fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K>
    where
        K: 'a;

    /// Get a reference to each part with the provided key along with its index.
    fn with_key<'a, 'b>(&'b self, key: &'a K) -> impl Iterator<Item = (usize, &'b V)> + 'a
    where
        'b: 'a,
        V: 'b;

    /// Gets a mutable reference to each part with the provided key along with its index.
    fn with_key_mut<'a, 'b>(
        &'b mut self,
        key: &'a K,
    ) -> impl Iterator<Item = (usize, &'b mut V)> + 'a
    where
        'b: 'a,
        V: 'b;
}

impl<I, K> Keyed<K, I> for MultipartInput<I, K>
where
    K: PartialEq,
{
    fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K>
    where
        K: 'a,
    {
        self.parts().iter().map(|(k, _)| k)
    }

    fn with_key<'a, 'b>(&'b self, key: &'a K) -> impl Iterator<Item = (usize, &'b I)> + 'a
    where
        'b: 'a,
        I: 'b,
    {
        self.parts()
            .iter()
            .enumerate()
            .filter_map(move |(i, (k, input))| (key == k).then_some((i, input)))
    }

    fn with_key_mut<'a, 'b>(
        &'b mut self,
        key: &'a K,
    ) -> impl Iterator<Item = (usize, &'b mut I)> + 'a
    where
        'b: 'a,
        I: 'b,
    {
        self.parts_mut()
            .iter_mut()
            .enumerate()
            .filter_map(move |(i, (k, input))| (key == k).then_some((i, input)))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::string::String;
    use std::{env, fs};

    use crate::{
        corpus::{Corpus, InMemoryOnDiskCorpus, Testcase},
        inputs::{BytesInput, Input, MultipartInput, ValueInput},
    };

    fn tmp_file(name: &str) -> std::path::PathBuf {
        let path = env::temp_dir().join(name);
        _ = fs::remove_file(&path);
        path
    }

    #[test]
    fn test_from_file_json() {
        let path = tmp_file("libafl_multipart_from_file_json.tmp");
        fs::write(
            &path,
            r#"[{"key":"keyA","value":"aAbBcC"},{"key":"keyB","value":""},{"key":"keyA","value":"ff"}]"#,
        )
        .unwrap();
        let input = MultipartInput::<BytesInput, String>::from_file(&path).unwrap();
        assert_eq!(input.len(), 3);
        let parts = input.parts();
        assert_eq!(parts[0].0, "keyA");
        assert_eq!(parts[0].1.as_ref().as_slice(), [0xaa, 0xbb, 0xcc]);
        assert_eq!(parts[1].0, "keyB");
        assert!(parts[1].1.as_ref().is_empty());
        assert_eq!(parts[2].0, "keyA");
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_from_file_json_invalid() {
        let path = tmp_file("libafl_multipart_from_file_json_invalid.tmp");

        fs::write(&path, r#"{"key":"keyA","value":"aabb"}"#).unwrap(); // object, not array
        assert!(MultipartInput::<BytesInput, String>::from_file(&path).is_err());

        fs::write(&path, r#"[{"key":"keyA"}]"#).unwrap(); // missing value
        assert!(MultipartInput::<BytesInput, String>::from_file(&path).is_err());

        fs::write(&path, r#"[{"key":"keyA","value":"abc"}]"#).unwrap(); // odd hex
        assert!(MultipartInput::<BytesInput, String>::from_file(&path).is_err());

        fs::write(&path, r#"[{"key":"keyA","value":"zz"}]"#).unwrap(); // invalid hex digit
        assert!(MultipartInput::<BytesInput, String>::from_file(&path).is_err());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_to_file_json_roundtrip() {
        let path = tmp_file("libafl_multipart_to_file_json.tmp");
        let input: MultipartInput<BytesInput, String> =
            MultipartInput::new(vec![("keyA".into(), BytesInput::new(vec![0xaa, 0xbb]))]);
        input.to_file(&path).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            r#"[{"key":"keyA","value":"aabb"}]"#
        );
        let loaded = MultipartInput::<BytesInput, String>::from_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.parts()[0].0, "keyA");
        assert_eq!(loaded.parts()[0].1.as_ref().as_slice(), [0xaa, 0xbb]);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_other_instantiation_keeps_postcard() {
        let path = tmp_file("libafl_multipart_postcard.tmp");
        let input: MultipartInput<ValueInput<u8>, String> =
            MultipartInput::new(vec![("a".into(), ValueInput::new(7_u8))]);
        input.to_file(&path).unwrap();
        // postcard, not JSON ...
        assert_ne!(fs::read(&path).unwrap().first(), Some(&b'['));
        // ... and it round-trips through the same trait methods.
        let loaded = MultipartInput::<ValueInput<u8>, String>::from_file(&path).unwrap();
        assert_eq!(loaded.parts()[0].0, "a");
        assert_eq!(loaded.parts()[0].1.into_inner(), 7);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_ondisk_corpus_speaks_json() {
        let dir = env::temp_dir().join("libafl_multipart_corpus_json");
        _ = fs::remove_dir_all(&dir);
        let mut corpus =
            InMemoryOnDiskCorpus::<MultipartInput<BytesInput, String>>::new(dir.clone()).unwrap();
        let input = MultipartInput::new(vec![("k".into(), BytesInput::new(vec![0xde, 0xad]))]);
        let id = corpus.add(Testcase::new(input)).unwrap();
        // the corpus wrote our JSON format (to_file via the trait, no shadowing possible) ...
        let file = corpus
            .get(id)
            .unwrap()
            .borrow()
            .file_path()
            .clone()
            .unwrap();
        assert_eq!(
            fs::read_to_string(&file).unwrap(),
            r#"[{"key":"k","value":"dead"}]"#
        );
        // ... and reads it back lazily (from_file via the trait).
        let tc_ref = corpus.get(id).unwrap();
        let mut tc = tc_ref.borrow_mut();
        let loaded = tc.load_input(&corpus).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.parts()[0].0, "k");
        assert_eq!(loaded.parts()[0].1.as_ref().as_slice(), [0xde, 0xad]);
        fs::remove_dir_all(&dir).unwrap();
    }
}
