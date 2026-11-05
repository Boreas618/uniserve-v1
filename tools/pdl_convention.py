"""Which operators append a `pdl` runtime word, and whether their package declares one.

A kernel learns the dependent-launch flag as a constexpr (it gates PTX waits at compile time) and the
driver learns it as a launch attribute from the module's metadata; it is not a runtime parameter. A
package whose kernel does take one and an operator that packs one are consistent with each other, and
an operator that packs a word the module does not declare shifts the two scratch addresses the
compiler's launcher appends after every kernel's parameters -- so this reports both sides.

Each operator gets one of four lines, and each `ok` carries its reason. An operator that packs no
dependent-launch word is `ok` whatever its module does -- no launcher's parameter list can shift, so
the flag is the module's own business. An operator that does pack one is `ok` when the module
declares it as a runtime parameter, `MISMATCH` when it does not, `uncovered` when no declared table
of the package parses the key (a manifest's answer, not a flag's), and `unknown` when the package's
`spec` states no signature -- which is the only shape of answer this checker cannot give.
A package's key reader ends an unusable key with `SystemExit`, so the parse is guarded at that
width: one table or one backend that cannot place the key must not end the run.
"""

import importlib.util
import pathlib
import re
import sys
import tomllib

OPS = pathlib.Path('crates/ops/src')
KERNELS = pathlib.Path('crates/ops/kernels')


def packs_pdl(operator):
    for path in OPS.rglob('*.rs'):
        text = path.read_text()
        if f'"{operator}"' not in text:
            continue
        for match in re.finditer(r"impl Op for (\w+) \{", text):
            depth = 0
            for index in range(match.end() - 1, len(text)):
                if text[index] == '{':
                    depth += 1
                elif text[index] == '}':
                    depth -= 1
                    if depth == 0:
                        body = text[match.start():index + 1]
                        break
            else:
                continue
            if f"const NAME: &'static str = \"{operator}\"" not in body:
                continue
            return '.flag(descriptor.pdl)' in body
    return None


def declares_pdl(operator, key):
    for manifest_path in KERNELS.glob(f'{operator}/*/package.toml'):
        source = manifest_path.parent / 'src'
        if not (source / 'build.py').exists():
            continue
        manifest = tomllib.loads(manifest_path.read_text())
        # A package may declare one descriptor table or several (`[[kernel]]`), one per shape it
        # serves; both forms describe the same fields, so the checker reads either.
        kernel = manifest.get('kernel', {})
        kernels = kernel if isinstance(kernel, list) else [kernel]
        # Each descriptor table is a shape the package serves, and a key belongs to exactly one of
        # them — merging the tables would let the second's `columns` values hide the first's, which is
        # how a key that parses under `columns = 64` reads as unparsable.
        tables = []
        for one in kernels:
            described = one.get('descriptors', {})
            for table in described if isinstance(described, list) else [described]:
                tables.append(table.get('dimensions', {}))
        sys.path.insert(0, str(source))
        try:
            sys.modules.pop('_compile', None)
            spec = importlib.util.spec_from_file_location('b', source / 'build.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import _compile
            placed = False
            for dimensions in tables:
                try:
                    fields = _compile.parse_key(key, dimensions)
                # A package's key reader ends a key it cannot place with `SystemExit`, which is
                # not an `Exception`: without the wider catch the checker stops at the first
                # table that does not fit and never reaches the one that does.
                except BaseException:
                    continue
                placed = True
                described = module.spec(fields, manifest.get('parameters', {}), 'sm_100a')
                return any(
                    name.lower() == 'pdl' and ty != 'constexpr'
                    for name, ty in described['signature'].items()
                )
            # No table of this package serves the key: that is a manifest's answer, not a flag's,
            # and it is not the same as a package this checker could not ask at all.
            return 'no-table' if not placed else None
        except BaseException:
            # One package that cannot be loaded or asked is not the run's answer: the key is
            # served by whichever backend declares it, and this one simply is not that backend.
            continue
        finally:
            sys.path.remove(str(source))
    return None


rows = []
declared_blocked = set()
# H3's dense rows are the deployment's own generator's file, which the coverage and store tools
# certify; `minimax-h3-keys.txt` is the older enumeration's walk print.
for fixture in ('notes/programs/qwen3-moe-keys.txt',
                'notes/programs/minimax-h3-deployment-keys.txt'):
    for line in pathlib.Path(fixture).read_text().split('\n'):
        # A fixture is rows plus a header. `#` carries the deployment, the walk's revision and the
        # rows no package declares yet — those are `# blocked:`, a tab, then operator, key and the
        # coverage tool's reason, and a row declared there has no package table to check against.
        if line.startswith('# blocked:\t'):
            _, operator, key, _ = line.split('\t', 3)
            declared_blocked.add((operator, key))
            continue
        if not line or line.startswith('#'):
            continue
        operator, key = line.split('\t')
        if (operator, key) in declared_blocked:
            continue
        rows.append((operator, key))

seen = set()
for operator, key in rows:
    if operator in seen:
        continue
    seen.add(operator)
    packs = packs_pdl(operator)
    declares = declares_pdl(operator, key)
    if packs is False:
        # The operator packs no dependent-launch word, so no launcher's parameter list can shift:
        # whatever the module does with the flag is the module's own business, not this question.
        print(f'ok       {operator:22} op packs False, so no launch parameter can shift')
        continue
    if declares == 'no-table':
        # No package's tables cover this key: the manifest owes a declaration, not a flag.
        print(f'uncovered {operator:22} op packs True  no declared table parses {key}')
        continue
    if declares is None:
        # A package was found but could not be asked (its `spec` states no signature) and the
        # operator does pack the word: this is the one shape of answer the checker cannot give.
        print(f'unknown  {operator:22} op packs True  the package states no signature')
        continue
    mark = 'MISMATCH' if declares is False else 'ok      '
    print(f'{mark} {operator:22} op packs {str(packs):5} module declares {declares}')
