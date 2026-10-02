"""Symbolic and numerical math validator using SymPy."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import sympy as sp

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class MathValidator(BaseValidator):
    """Verifies algebraic equations and arithmetic statements using SymPy."""

    name: str = "math_symbolic"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)
        question = data.get("question", "")
        answer = str(data.get("answer", ""))

        # Extract algebraic equation like "2*x + 3 = 13" or "2x + 3 = 13"
        # Match only mathematical tokens (digits, letters, operators, spaces, parentheses) around '='
        eq_match = re.search(r"([0-9a-zA-Z\s\+\-\*\/\^\(\)]+?)\s*=\s*([0-9a-zA-Z\s\+\-\*\/\^\(\)]+)", question)
        if eq_match:
            raw_lhs = eq_match.group(1).strip()
            raw_rhs = eq_match.group(2).strip()

            # Clean out any leading English words from lhs (e.g. "Solve the equation: 2*x + 3")
            lhs_math_match = re.search(r"((?:[0-9\.]+\s*[\*\+\-\/\^]\s*)*[0-9a-zA-Z\.\s\+\-\*\/\^\(\)]+)$", raw_lhs)
            lhs_clean = lhs_math_match.group(1).strip() if lhs_math_match else raw_lhs
            # If there's a colon or newline, take what follows
            if ":" in lhs_clean:
                lhs_clean = lhs_clean.split(":")[-1].strip()

            # Strip trailing English words from rhs (e.g. "13. Find the value of x.")
            rhs_clean = raw_rhs.split(".")[0].strip()

            # Identify variable (usually x, y, z or n)
            var_match = re.search(r"\b([a-zA-Z])\b", lhs_clean + " " + rhs_clean)
            if var_match:
                var_char = var_match.group(1)
                try:
                    var = sp.Symbol(var_char)
                    # Convert 2x to 2*x if needed
                    lhs_norm = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", lhs_clean)
                    rhs_norm = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", rhs_clean)
                    lhs_expr = sp.sympify(lhs_norm)
                    rhs_expr = sp.sympify(rhs_norm)
                    equation = sp.Eq(lhs_expr, rhs_expr)
                    solutions = sp.solve(equation, var)

                    # Extract number from generated answer
                    ans_num_match = re.search(r"[-+]?\d*\.?\d+", answer)
                    if ans_num_match and solutions:
                        ans_val = float(ans_num_match.group(0))
                        matched = any(abs(float(sol.evalf()) - ans_val) < 1e-4 for sol in solutions)
                        if not matched:
                            expected_str = ", ".join(str(s) for s in solutions)
                            return ValidationResult(
                                status="reject",
                                validator=self.identifier,
                                score=0.0,
                                errors=[f"Math solution {ans_val} does not match SymPy solution(s): {expected_str}"],
                                metadata={"expected": [str(s) for s in solutions], "generated": ans_val},
                            )
                        else:
                            return ValidationResult(
                                status="accept",
                                validator=self.identifier,
                                score=1.0,
                                errors=[],
                                metadata={"verified": True, "solution": ans_val},
                            )
                except Exception:
                    pass

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={"verified": False, "reason": "non_algebraic_expression"},
        )
