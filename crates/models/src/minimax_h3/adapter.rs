//! FastVideo's adapter container, `fastvideo-lora-v2`: the form a FastH3 student ships in as
//! edits of the base release's network, which the family fuses at the loading boundary
//! ([`nn::adapter`]).
//!
//! A deployment's adapter directory holds the container as `adapter_model.safetensors`, beside
//! the student's contract ([`crate::minimax_h3::config::Contract`]), which states the network the
//! adapter edits, its schedule and its attention. Each tensor's name is the edited module's path in
//! the network's own namespace with one of five suffixes, and the suffix names the parameter it
//! edits and how:
//!
//! | Adapter tensor | Edits | Role |
//! | --- | --- | --- |
//! | `<module>.lora_A.weight` `[rank, in]`, `<module>.lora_B.weight` `[out, rank]` | `<module>.weight` | down, up |
//! | `<module>.diff` | `<module>.weight` | full-rank delta |
//! | `<module>.diff_b` | `<module>.bias` | full-rank delta |
//! | `<module>.set_weight` | `<module>.weight` | assigned: the base does not hold it |
//!
//! The header metadata states the container (`format`), the reconstruction it encodes
//! (`W = W_base + lora_B @ lora_A; then .diff/.diff_b added and .set_weight assigned`, with no
//! `alpha`: a scale of one), how many tensors of each kind the writer emitted
//! (`low_rank_tensors`, `diff_tensors`, `set_weight_tensors`) and the factors' `rank`.

use std::collections::BTreeMap;

use nn::adapter::{Contents, Format, Role};

/// FastVideo's adapter container.
pub struct FastVideoLora;

impl FastVideoLora {
    /// The container's name, as its metadata's `format` states it.
    pub const FORMAT: &'static str = "fastvideo-lora-v2";

    /// The file an adapter directory holds the container as.
    pub const FILE: &'static str = "adapter_model.safetensors";

    /// Each suffix of the container, with the leaf of the parameter it edits and its role.
    /// `.diff_b` is matched before `.diff`, whose name it ends with.
    const SUFFIXES: [(&'static str, &'static str, Role); 5] = [
        (".lora_A.weight", "weight", Role::Down),
        (".lora_B.weight", "weight", Role::Up),
        (".diff_b", "bias", Role::Diff),
        (".diff", "weight", Role::Diff),
        (".set_weight", "weight", Role::Assign),
    ];
}

impl Format for FastVideoLora {
    fn file(&self) -> &'static str {
        Self::FILE
    }

    fn role(&self, name: &str) -> Option<(String, Role)> {
        Self::SUFFIXES.iter().find_map(|(suffix, leaf, role)| {
            name.strip_suffix(suffix)
                .map(|module| (format!("{module}.{leaf}"), *role))
        })
    }

    /// The writer's statement is the container's, and the file must carry each count it
    /// states and the one rank it states: a truncated file is otherwise indistinguishable from
    /// a small one.
    fn check(
        &self,
        metadata: &BTreeMap<String, String>,
        contents: &Contents,
    ) -> std::result::Result<(), String> {
        match metadata.get("format") {
            Some(format) if format == Self::FORMAT => {}
            stated => {
                return Err(format!(
                    "the adapter states the format {stated:?}; this family fuses {} adapters, \
                     whose reconstruction it implements",
                    Self::FORMAT
                ));
            }
        }

        let count = |key: &str| -> std::result::Result<u64, String> {
            let stated = metadata
                .get(key)
                .ok_or_else(|| format!("the adapter does not state `{key}`"))?;
            stated
                .parse()
                .map_err(|_| format!("`{key}` is {stated:?}, not a count"))
        };
        for (key, carried) in [
            ("low_rank_tensors", contents.low_rank),
            ("diff_tensors", contents.diffs),
            ("set_weight_tensors", contents.assigned),
        ] {
            let stated = count(key)?;
            if stated != carried {
                return Err(format!(
                    "the adapter states {key} = {stated} and carries {carried}: it is incomplete"
                ));
            }
        }

        if metadata.contains_key("rank") {
            let rank = count("rank")?;
            if let Some(other) = contents.ranks.iter().find(|carried| **carried != rank) {
                return Err(format!(
                    "the adapter's factors have rank {other}, and it states rank {rank}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// The header of a rank-64 adapter that carries `counts` low-rank, diff and assigned
    /// tensors.
    fn metadata(counts: [&str; 3]) -> BTreeMap<String, String> {
        [
            ("format", FastVideoLora::FORMAT),
            ("rank", "64"),
            ("low_rank_tensors", counts[0]),
            ("diff_tensors", counts[1]),
            ("set_weight_tensors", counts[2]),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
    }

    /// What a file of two factors of rank `rank`, three deltas and one assignment carries.
    fn contents(rank: u64) -> Contents {
        Contents {
            low_rank: 2,
            diffs: 3,
            assigned: 1,
            ranks: BTreeSet::from([rank]),
        }
    }

    #[test]
    fn each_suffix_names_the_parameter_it_edits_and_how() {
        let role = |name: &str| FastVideoLora.role(name);

        assert_eq!(
            role("blocks.0.attn.to_q.lora_A.weight"),
            Some(("blocks.0.attn.to_q.weight".to_owned(), Role::Down))
        );
        assert_eq!(
            role("blocks.0.attn.to_q.lora_B.weight"),
            Some(("blocks.0.attn.to_q.weight".to_owned(), Role::Up))
        );
        assert_eq!(
            role("proj_out.diff_b"),
            Some(("proj_out.bias".to_owned(), Role::Diff))
        );
        assert_eq!(
            role("norm.diff"),
            Some(("norm.weight".to_owned(), Role::Diff))
        );
        assert_eq!(
            role("blocks.0.attn.to_gate_compress.set_weight"),
            Some((
                "blocks.0.attn.to_gate_compress.weight".to_owned(),
                Role::Assign
            ))
        );
        assert_eq!(role("blocks.0.attn.to_q.weight"), None);
    }

    #[test]
    fn a_file_carries_what_its_writer_states() {
        let stated = metadata(["2", "3", "1"]);
        assert_eq!(FastVideoLora.check(&stated, &contents(64)), Ok(()));

        let error = FastVideoLora
            .check(&metadata(["2", "2", "1"]), &contents(64))
            .unwrap_err();
        assert!(error.contains("diff_tensors"), "{error}");

        let error = FastVideoLora.check(&stated, &contents(32)).unwrap_err();
        assert!(error.contains("rank 32"), "{error}");

        let mut foreign = stated.clone();
        foreign.insert("format".to_owned(), "peft".to_owned());
        let error = FastVideoLora.check(&foreign, &contents(64)).unwrap_err();
        assert!(error.contains("peft"), "{error}");
    }
}
