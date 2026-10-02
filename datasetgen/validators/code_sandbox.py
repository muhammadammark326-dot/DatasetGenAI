"""Sandboxed code execution validator with AST analysis and subprocess timeouts."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class DangerousCodeVisitor(ast.NodeVisitor):
    """AST visitor detecting potentially dangerous calls or imports."""

    BLOCKED_MODULES: Set[str] = {
        "os", "subprocess", "sys", "shutil", "socket", "urllib", "requests",
        "pty", "ctypes", "winreg", "posix", "pathlib", "importlib", "builtins"
    }
    BLOCKED_FUNCTIONS: Set[str] = {
        "eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "delattr"
    }

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in self.BLOCKED_MODULES:
                self.violations.append(f"Disallowed import of module '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in self.BLOCKED_MODULES:
                self.violations.append(f"Disallowed import from module '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.BLOCKED_FUNCTIONS:
                self.violations.append(f"Disallowed call to restricted function '{node.func.id}()'")
        self.generic_visit(node)


class CodeSandboxValidator(BaseValidator):
    """Verifies that generated code compiles and passes assertions in a safe sandbox."""

    name: str = "code_sandbox"
    version: str = "v1"

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = timeout_seconds

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)

        code_snippet = data.get("code") or data.get("solution") or data.get("function")
        test_snippet = data.get("test_cases") or data.get("test") or data.get("assertions") or ""

        if not code_snippet:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=["Code example missing 'code' or 'solution' field."],
            )

        full_script = f"{code_snippet}\n\n{test_snippet}"

        # 1. AST Syntax Check
        try:
            tree = ast.parse(full_script)
        except SyntaxError as e:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=[f"Code syntax error: {e.msg} at line {e.lineno}"],
                metadata={"syntax_error": True},
            )

        # 2. Dangerous Calls & Imports Security Check
        visitor = DangerousCodeVisitor()
        visitor.visit(tree)
        if visitor.violations:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=[f"Security violation in code: {v}" for v in visitor.violations],
                metadata={"security_failed": True, "violations": visitor.violations},
            )

        # 3. Subprocess Execution Sandbox
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sandbox_test.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmpdir,
                )
                if result.returncode != 0:
                    err_msg = result.stderr.strip().splitlines()[-1] if result.stderr else "Unknown error"
                    return ValidationResult(
                        status="reject",
                        validator=self.identifier,
                        score=0.0,
                        errors=[f"Runtime assertion failure: {err_msg}"],
                        metadata={"stderr": result.stderr[:300], "exit_code": result.returncode},
                    )
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    status="reject",
                    validator=self.identifier,
                    score=0.0,
                    errors=[f"Execution timed out (> {self.timeout_seconds}s)"],
                    metadata={"timeout": True},
                )
            except Exception as e:
                return ValidationResult(
                    status="reject",
                    validator=self.identifier,
                    score=0.0,
                    errors=[f"Sandbox execution error: {str(e)}"],
                )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={"sandbox_passed": True, "tested_assertions": bool(test_snippet)},
        )
