#!/usr/bin/env python3
"""Mutation tests of the Rust buffer pool (L4): every deleted rule is caught.

    mutations.py [RULE...]          (R1 to R10 by default)

For each rule R1 to R10 of specs/buffer-pool.md §9, the guard that enforces
it in crates/storage-server/src/buffer_pool is replaced as stated below, the
crate's buffer pool tests are run, and the source is restored. A rule is
caught when the mutated crate compiles and at least one test fails. Prints
one line per rule with the tests that caught it and exits non-zero when a
rule is not caught, a mutation no longer applies (its text is not found
exactly once), or the mutated crate does not compile.

Run it with nothing else building the crate: it edits the source in place.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/storage-server/src/buffer_pool"

# rule: what is left out, and the replacements (file, text, replacement).
MUTATIONS = {
    "R1": ("an allocation at an occupied position is taken", [(
        "mod.rs",
        """        if count == 0 || !self.free_positions(client, owner, class, at, count) {
            return Err(Cause::Position);
        }
        let regions =""",
        """        let regions =""",
    )]),
    "R2": ("runs return to their region while other holders remain", [(
        "mod.rs",
        """        if let Ok(block) = Rc::try_unwrap(block) {
            for run in block.runs {""",
        """        {
            for run in block.runs.iter().map(Run::again) {""",
    ), (
        "region.rs",
        """    /// Splits off the first `count` units; the rest stays in `self`.""",
        """    pub(super) fn again(&self) -> Run {
        Run { region: self.region, first: self.first, count: self.count }
    }

    /// Splits off the first `count` units; the rest stays in `self`.""",
    )]),
    "R3": ("stamped work takes effect when it is taken", [(
        "mod.rs",
        """                && found.completed < *stamp
""",
        """                && found.completed < *stamp
                && false
""",
    ), (
        "mod.rs",
        """                && found.completed < stamp
""",
        """                && found.completed < stamp
                && false
""",
    )]),
    "R4": ("a second block is committed under a cached key", [(
        "mod.rs",
        """        if cached {
            self.reclaim(block);
            return Completion::Duplicate;""",
        """        if cached && false {
            self.reclaim(block);
            return Completion::Duplicate;""",
    )]),
    "R5": ("a clone hands out the cached block as private", [(
        "copies.rs",
        """        let region = source.region();
        let size = source.units();
""",
        """        if let Some(found) = self.owners.get_mut(&(client, owner)) {
            found.tables[usize::from(class)].insert(at, Held::Private(Rc::clone(&source)));
            return Ok(Some(Completion::Cloned));
        }
        let region = source.region();
        let size = source.units();
""",
    )]),
    "R6": ("a commit is cancelled when its owner let go of the block", [(
        "mod.rs",
        """        // R4: a key that is cached keeps its block; this one stays private.
        if cached {""",
        """        let holds = self.owners.get(&(client, owner)).is_some_and(|found| {
            let held = found.tables[usize::from(class)].get(&at);
            matches!(held, Some(Held::Private(other)) if Rc::ptr_eq(other, &block))
        });
        if !holds {
            self.reclaim(block);
            return Completion::Duplicate;
        }
        if cached {""",
    )]),
    "R7": ("an eviction takes a block that has a pin", [(
        "mod.rs",
        """block.region() == region && Rc::strong_count(block) == 1;""",
        """block.region() == region;""",
    )]),
    "R8": ("a copy made on trial is visible to every epoch", [(
        "mod.rs",
        """        maker.is_none_or(|maker| self.sessions.get(&client) == Some(&maker))""",
        """        true || maker.is_none_or(|maker| self.sessions.get(&client) == Some(&maker))""",
    )]),
    "R9": ("a task's blocks are released at its deadline", [(
        "tasks.rs",
        """            self.stopped.push(task);
            if let Some(completion) = completion {""",
        """            for copy in self.copies.remove(&task).unwrap_or_default() {
                for block in copy.reads {
                    self.reclaim(block);
                }
                self.reclaim(copy.write);
            }
            self.stopped.push(task);
            if let Some(completion) = completion {""",
    )]),
    "R10": ("a unit joins a region whatever holds its bytes", [(
        "mod.rs",
        """                remove_range(quota, offset, bytes)""",
        """                remove_range(quota, offset, bytes) || true""",
    )]),
}


def run_tests():
    command = ["cargo", "test", "-p", "storage-server", "--lib", "buffer_pool"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    compiled = "could not compile" not in output
    failed = re.findall(r"^test buffer_pool::(\S+) \.\.\. FAILED", output, flags=re.M)
    return compiled, failed, output


def main():
    rules = sys.argv[1:] or list(MUTATIONS)
    status = 0
    for rule in rules:
        left_out, replacements = MUTATIONS[rule]
        originals = {name: (SOURCE / name).read_text() for name, _, _ in replacements}
        try:
            texts = dict(originals)
            stale = False
            for name, text, replacement in replacements:
                stale |= texts[name].count(text) != 1
                texts[name] = texts[name].replace(text, replacement)
            if stale:
                print(f"FAIL  {rule}: the mutation no longer applies ({left_out})")
                status = 1
                continue
            for name, text in texts.items():
                (SOURCE / name).write_text(text)
            compiled, failed, output = run_tests()
        finally:
            for name, text in originals.items():
                (SOURCE / name).write_text(text)

        if not compiled:
            errors = [line for line in output.splitlines() if line.startswith("error")]
            print(f"FAIL  {rule}: the mutated crate does not compile: {errors[:1]}")
            status = 1
        elif failed:
            print(f"ok    {rule}: {left_out}; caught by {', '.join(sorted(failed))}")
        else:
            print(f"FAIL  {rule}: {left_out}; no test fails")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
