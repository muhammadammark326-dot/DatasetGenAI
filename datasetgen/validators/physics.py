"""Physics numeric validator using SymPy and Pint."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple, Union

import sympy as sp

from datasetgen.schemas.blueprint import Blueprint
from datasetgen.schemas.example import GeneratedExample
from datasetgen.schemas.validation import ValidationResult
from datasetgen.validators.base import BaseValidator


class PhysicsNumericValidator(BaseValidator):
    """Recomputes physics numerical answers independently with SymPy to verify truth."""

    name: str = "physics_numeric"
    version: str = "v1"

    def validate(
        self,
        example: Union[Dict[str, Any], GeneratedExample],
        blueprint: Blueprint,
    ) -> ValidationResult:
        data = self._extract_data(example)
        question = data.get("question", "")
        answer = str(data.get("answer", ""))
        topic = data.get("topic", "").lower()

        gen_num = self._extract_first_number(answer)
        if gen_num is None:
            return ValidationResult(
                status="accept",
                validator=self.identifier,
                score=1.0,
                errors=[],
                metadata={"type": "conceptual"},
            )

        expected_num, expected_unit, formula = self._recompute_expected(question, topic)

        if expected_num is None:
            return ValidationResult(
                status="accept",
                validator=self.identifier,
                score=1.0,
                errors=[],
                metadata={"verified": False, "reason": "no_matching_formula"},
            )

        # Check numerical agreement within 2% tolerance
        rel_diff = abs(gen_num - expected_num) / max(abs(expected_num), 1e-6)
        if rel_diff > 0.02:
            return ValidationResult(
                status="reject",
                validator=self.identifier,
                score=0.0,
                errors=[
                    f"Computed answer ({expected_num:.2f}) does not match generated answer ({gen_num:.2f}) for formula {formula}"
                ],
                metadata={
                    "expected": round(expected_num, 4),
                    "generated": round(gen_num, 4),
                    "relative_error": round(rel_diff, 4),
                    "formula": formula,
                },
            )

        return ValidationResult(
            status="accept",
            validator=self.identifier,
            score=1.0,
            errors=[],
            metadata={
                "expected": round(expected_num, 4),
                "generated": round(gen_num, 4),
                "formula": formula,
                "verified": True,
            },
        )

    def _extract_first_number(self, text: str) -> Optional[float]:
        """Extract first float or int from string."""
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None

    def _recompute_expected(
        self, question: str, topic: str
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Identify physical relationship and calculate the exact ground truth with SymPy."""
        # 1. Newton's 2nd Law: F = m * a
        m_mass = re.search(r"(\d+\.?\d*)\s*kg\b", question, re.I)
        m_acc = re.search(r"(\d+\.?\d*)\s*m/s\^?2", question, re.I)
        if m_mass and m_acc and ("force" in question.lower() or "net force" in question.lower()):
            m = float(m_mass.group(1))
            a = float(m_acc.group(1))
            f_sym = sp.Symbol("F")
            sol = sp.solve(sp.Eq(f_sym, m * a), f_sym)
            if sol:
                return float(sol[0]), "N", "F = m * a"

        # 2. Ohm's Law: I = V / R or V = I * R
        m_res = re.search(r"(\d+\.?\d*)\s*(?:ohms|Ω)", question, re.I)
        m_volt = re.search(r"(\d+\.?\d*)\s*v(?:olts)?\b", question, re.I)
        if m_res and m_volt and ("current" in question.lower() or "amperes" in question.lower()):
            r = float(m_res.group(1))
            v = float(m_volt.group(1))
            i_sym = sp.Symbol("I")
            sol = sp.solve(sp.Eq(i_sym, v / r), i_sym)
            if sol:
                return float(sol[0]), "A", "I = V / R"

        # 3. Wave equation: v = f * lambda
        m_freq = re.search(r"(\d+\.?\d*)\s*hz\b", question, re.I)
        m_wave = re.search(r"(\d+\.?\d*)\s*m\b", question, re.I)
        if m_freq and m_wave and ("speed" in question.lower() or "velocity" in question.lower()):
            f = float(m_freq.group(1))
            lam = float(m_wave.group(1))
            v_sym = sp.Symbol("v")
            sol = sp.solve(sp.Eq(v_sym, f * lam), v_sym)
            if sol:
                return float(sol[0]), "m/s", "v = f * lambda"

        # 4. Heat energy: Q = m * c * DeltaT (for water c = 4184 J/(kg K))
        m_water = re.search(r"(\d+\.?\d*)\s*kg", question, re.I)
        m_temp = re.search(r"(\d+\.?\d*)\s*k(?:elvin)?\b", question, re.I)
        if m_water and m_temp and ("heat" in question.lower() or "energy" in question.lower()):
            m = float(m_water.group(1))
            dt = float(m_temp.group(1))
            q_sym = sp.Symbol("Q")
            sol = sp.solve(sp.Eq(q_sym, m * 4184 * dt), q_sym)
            if sol:
                return float(sol[0]), "J", "Q = m * c * DeltaT"

        # 5. Thin Lens Equation: 1/f = 1/do + 1/di
        m_do = re.search(r"placed\s*(\d+\.?\d*)\s*cm", question, re.I)
        m_f = re.search(r"focal\s*length\s*(\d+\.?\d*)\s*cm", question, re.I)
        if m_do and m_f and "image" in question.lower():
            do = float(m_do.group(1))
            focal = float(m_f.group(1))
            if do != focal:
                di_sym = sp.Symbol("di")
                sol = sp.solve(sp.Eq(1 / focal, 1 / do + 1 / di_sym), di_sym)
                if sol:
                    return float(sol[0]), "cm", "1/f = 1/do + 1/di"

        return None, None, None
