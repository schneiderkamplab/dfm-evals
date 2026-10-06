"""Standalone prompts and judgment parsing; standard-library dependencies only."""

import re

JUDGE_TEMPLATE_DA = """\
Du vurderer et indsendt svar mod et ekspertsvar om en dansk talemåde.

[BEGIN DATA]
************
[Talemåde]: {talemaade_udtryk}
************
[Facit]: {criterion}
************
[Modelsvar]: {answer}
************
[END DATA]

Sammenlign det semantiske indhold i modelsvaret med facit. \
Ignorer forskelle i stil, grammatik eller tegnsætning.

I denne opgave skal modelsvaret faktisk forklare, hvad talemåden betyder. \
Det er ikke nok blot at gentage talemåden, omskrive den næsten ordret eller \
bruge den i en ny sætning. Svar kun positivt, hvis modelsvaret tydeligt \
udtrykker selve betydningen eller definitionen af talemåden med egne ord.

Indeholder modelsvaret den korrekte betydning af talemåden?

{instructions}
"""

JUDGE_INSTRUCTIONS_DA = (
    "Forklar først kort og trin for trin, hvordan modelsvaret stemmer overens "
    "med facit. Skriv derefter præcis én afsluttende linje med formatet "
    "'GRADE: $LETTER' hvor LETTER er en af CPI. Brug ikke 'GRADE:' tidligere "
    "i svaret, og placer grade-linjen som den allersidste linje. Et svar er "
    "kun korrekt, hvis det faktisk forklarer betydningen af talemåden. Det er "
    "ikke nok at gentage udtrykket, omskrive det næsten ordret eller give en "
    "cirkulær parafrase uden at skrive, hvad talemåden betyder. "
    'Vælg én mulighed: "C" for korrekte svar, '
    '"P" for delvist korrekte svar, '
    'eller "I" for forkerte svar.'
)

JUDGE_INSTRUCTIONS_V2 = (
    "Skriv kun præcis én linje: GRADE: C, GRADE: P eller GRADE: I. "
    "Ingen forklaring eller anden tekst. Et svar er "
    "kun korrekt, hvis det faktisk forklarer betydningen af talemåden. Det er "
    "ikke nok at gentage udtrykket, omskrive det næsten ordret eller give en "
    "cirkulær parafrase uden at skrive, hvad talemåden betyder. "
    'Vælg én mulighed: "C" for korrekte svar, '
    '"P" for delvist korrekte svar, '
    'eller "I" for forkerte svar.'
)


class InvalidJudgmentError(ValueError):
    """The judge did not return a complete, valid judgment (not an I grade)."""


def build_judge_prompt(*, talemaade_udtryk: str, criterion: str, answer: str) -> str:
    """Use the original semantic template, changing only output instructions."""
    return JUDGE_TEMPLATE_DA.format(
        talemaade_udtryk=talemaade_udtryk,
        criterion=criterion,
        answer=answer,
        instructions=JUDGE_INSTRUCTIONS_V2,
    )


def parse_judgment(text: str) -> float:
    """Parse exactly one uppercase grade; reject explanations and ambiguous output."""
    match = re.fullmatch(r"GRADE: ([CPI])", text.strip())
    if match is None:
        raise InvalidJudgmentError("Expected exactly GRADE: C, GRADE: P or GRADE: I")
    return {"C": 1.0, "P": 0.5, "I": 0.0}[match[1]]
