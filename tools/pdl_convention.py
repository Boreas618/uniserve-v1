"""Which operators append a `pdl` runtime word, and whether their package declares one.

A kernel learns the dependent-launch flag as a constexpr (it gates PTX waits at compile time) and the
driver learns it as a launch attribute from the module's metadata; it is not a runtime parameter. A
package whose kernel does take one and an operator that packs one are consistent with each other, and
an operator that packs a word the module does not declare shifts the two scratch addresses the
compiler's launcher appends after every kernel's parameters -- so this reports both sides.
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
        dimensions = manifest.get('kernel', {}).get('descriptors', {}).get('dimensions', {})
        sys.path.insert(0, str(source))
        try:
            sys.modules.pop('_compile', None)
            spec = importlib.util.spec_from_file_location('b', source / 'build.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import _compile
            fields = _compile.parse_key(key, dimensions)
            described = module.spec(fields, manifest.get('parameters', {}), 'sm_100a')
            return any(name.lower() == 'pdl' and ty != 'constexpr' for name, ty in described['signature'].items())
        except Exception:
            return None
        finally:
            sys.path.remove(str(source))
    return None


rows = []
for fixture in ('notes/programs/qwen3-moe-keys.txt', 'notes/programs/minimax-h3-keys.txt'):
    for line in pathlib.Path(fixture).read_text().strip().split('\n'):
        operator, key = line.split('\t')
        rows.append((operator, key))

seen = set()
for operator, key in rows:
    if operator in seen:
        continue
    seen.add(operator)
    packs = packs_pdl(operator)
    declares = declares_pdl(operator, key)
    mark = 'MISMATCH' if packs and declares is False else 'ok      '
    print(f'{mark} {operator:22} op packs {str(packs):5} module declares {declares}')
