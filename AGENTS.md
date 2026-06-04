# AGENTS.md

This file defines mandatory architecture and coding rules for AI coding agents.

Agents must follow these rules unless explicitly instructed otherwise.

---

# General Philosophy

- Code must be readable, explicit, and maintainable.
- Prefer simplicity over clever abstractions.
- Follow **moderate SOLID principles**.
- Prefer composition over inheritance.
- Business architecture must remain clear from the directory structure.
- Reflect changes made in README and docs, keep it clean descriptive of the current state of repo, dont do "version control" in readme or docs

---

# Mandatory Documentation (Docstrings)

Every **class, function, and method MUST contain a docstring**.

Docstrings should explain:

- purpose
- important behavior
- parameters when relevant
- return values when not obvious

Example:

```python
class ServiceSettings(BaseModel):
    """Settings of the service runtime."""
````

Rules:

* public classes must have docstrings
* public functions must have docstrings
* public methods must have docstrings

Prefer **short and clear descriptions**.

---

# Business Module Architecture

Code must be organized around **business capabilities**, not technical layers.

Example:

```
src/
    code_wiki/
    repo_monitoring/
    adapters/
    database/
    config.py
    main.py
```

Rules:

* each capability has its own package
* package names reflect the business domain
* avoid generic folders like `services` or `core`

Good:

```
src/code_wiki/
src/repo_monitoring/
```

Bad:

```
src/services/
src/core/
```

---

# Internal Module Structure

Example agent module:

```
src/code_wiki/
    ast_node/
    cluster_node/
    generation_node/
    prompts/
    utils/
    doc_generator_graph.py
    runtime_config.py
```

Rules:

* structure must reflect responsibilities
* avoid giant modules
* prefer small focused packages

---

# LangGraph Conventions

## Graph assembly

Graph orchestration must live in:

```
*_graph.py
```

Example:

```
doc_generator_graph.py
```

Responsibilities:

* state schema
* context schema
* routing logic
* node registration
* graph compilation

Graph files must stay **readable orchestration layers**.

---

## Node organization

Nodes must represent **one step of the pipeline**.

Example:

```
ast_node/
cluster_node/
generation_node/
```

Rules:

* one node = one responsibility
* move helpers to `utils`

---

# Context and State

Runtime context must use **dataclasses**.

Example:

```python
@dataclass(frozen=True, slots=True, repr=True)
class BaseCtx:
    settings: Settings
    llm_client: ChatOpenAI
```

Rules:

* context objects should be immutable
* dependencies grouped in context

For stable schemas prefer **Pydantic models**.

Bad:

```python
Mapping[str, Any]
```

Better:

```python
class ModuleTree(BaseModel):
    nodes: list[ModuleNode]
```

---

# Central Composition

Application wiring must happen in the **composition root**.

Usually:

```
src/main.py
```

Responsibilities:

* load config
* create dependencies
* initialize graphs
* connect adapters
* wire business modules

---

# Adapter Layer

Adapters connect protocols to business logic.

Example:

```
src/adapters/
    continue_adapter/
```

Adapters may include:

* HTTP routers
* webhooks
* workers
* integration bridges

Rules:

* adapters may import business modules
* business modules must not import adapters

---

# Imports

Use **absolute imports only**.

Correct:

```python
from src.code_wiki.doc_generator_graph import build_graph
```

Wrong:

```python
from ..doc_generator_graph import build_graph
```

`__init__.py` files must stay minimal.

Rules:

* do not use `__init__.py` as an export barrel
* do not re-export symbols from package internals via `__init__.py`
* import from the concrete module where the symbol is defined
* do not use nested imports inside functions or methods; imports must be module-level
* avoid quoted forward-reference annotations; import concrete types at module level when practical
* `__init__.py` may contain only a short package docstring or package marker code when truly necessary

---

# Comments and Docstrings Language

Rules:

* comments and docstrings in application code must be written in Russian
* keep English only for external protocol names, library names, literal values, or quoted third-party messages

---

# Dataclasses

Default dataclass pattern:

```python
@dataclass(frozen=True, slots=True, repr=True)
class Foo:
    pass
```

Mutable case:

```python
@dataclass(slots=True, repr=True)
class Foo2:
    pass
```

Rules:

* prefer `frozen=True`
* always `slots=True`
* always `repr=True`

---

# Typing

Prefer **collections.abc**.

```python
from collections.abc import Iterable, Mapping, Sequence
```

Built-in generics:

```
list[str]
dict[str, int]
```

are allowed only when appropriate.

Rules:

* prefer `collections.abc`
* avoid `typing.List`, `typing.Dict`
* avoid vague container types

Bad:

```
Mapping[str, Any]
```

Better:

```
Mapping[str, ModuleInfo]
```

For structured data prefer **Pydantic models**.

---

# Static Typing (mypy)

The project uses **mypy**.

Rules:

* code should pass mypy
* fix typing issues when modifying code
* avoid `Any`
* avoid `# type: ignore`

---

# Async Rules

Prefer async functions for IO-bound operations.

Rules:

* avoid blocking IO inside async code
* prefer async clients
* avoid synchronous network libraries

---

# Configuration Architecture

Use **nested configuration models**.

Example:

```python
class Settings(BaseSettings):
    service: ServiceSettings
    llm: LLMSettings
    db: DatabaseSettings
```

Rules:

* avoid flat configs
* each section has its own model
* use Field(..., description=...)

---

# Configuration Initialization

Use cached settings factory.

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Config is injected via:

```
app.state.settings
```

---

# Paths

Prefer `pathlib`.

Good:

```
Path("data/file.txt")
```

Avoid:

```
os.path
```

---

# Database and Migrations

Use **Alembic**.

Rules:

* LLM must NOT invent migration logic
* modify SQLAlchemy models only
* migrations generated via:

```
alembic revision --autogenerate
```

PostgreSQL enum types must use **explicit, domain-specific names**.

Rules:

* do not rely on auto-derived enum type names
* do not reuse or mirror the column name as the database enum type name
* enum type names must describe the business context, for example `docgen_run_status_enum`
* when PostgreSQL schema is used, enum types must be created in that schema explicitly

---

# Logging

Use module-level logger.

```
logger = logging.getLogger(__name__)
```

Use one-line logger calls with `%s` placeholders and inline `key=value` pairs.

Rules:

* keep each logger call in one physical line
* never split a `logger.*(...)` call across multiple physical lines, even when the message is long
* use printf-style placeholders (`%s`) instead of f-strings in logger arguments
* include context in the message body as `key=value` pairs
* do not use `extra=...` for business logs
* do not use context wrappers/filters/adapters for log enrichment

Example:

```python
logger.warning("Не удалось определить service_version: repo_root=%s git_dir=%s fallback=%s", resolved_root, git_dir, fallback)
```

---

# Prompt Organization

All prompts must live in:

```
prompts/
```

Rules:

* prompts stored as `.txt`
* do not inline prompts in code
* prompts must be reusable

Example:

```
prompts/
    cluster_system.txt
    generate_docs_system.txt
```

Load prompts from files.

---

# Forbidden Patterns

Avoid:

* relative imports
* global mutable state
* giant service modules
* inline prompts
* blocking IO in async
* manual alembic migrations
* excessive Any
* excessive type ignore
* unsolicited workaround logic (do only what was explicitly requested)
* suppressing warnings/errors without explicit request
* redundant env names like `DB__DB_NAME` when `DB__NAME` is sufficient
* overengineered alias chains for config fields without clear migration requirement
* nested imports inside functions or methods
* quoted type annotations when direct imports are practical

---

# Change Minimalism

Rules:

* implement only requested behavior and required technical consequences
* if a workaround is optional, ask before adding it
* prefer one canonical config name over multiple aliases
* keep naming straightforward (`db.name`, `db.schema`) unless user asks otherwise

---

# Code Style Summary

Prefer:

* business-capability modules
* dataclasses
* pathlib
* collections.abc
* pydantic schemas
* docstrings everywhere
* class-based builders for infrastructure wiring such as Redis, checkpointers, external clients, and lifecycle resources
* SOLID structure
* safe collection access: before indexing a list/array (`items[0]`), explicitly ensure it is non-empty

---

# Agent Instruction

When generating code:

1. follow this file
2. preserve architecture
3. keep modules readable
4. add docstrings
5. respect business modules
6. keep graphs readable
7. avoid inventing migrations
8. improve typing
9. prefer collections.abc
10. store prompts in `.txt`
