import logging
import re
import signal
from importlib.metadata import version
from typing import Dict, List, Optional

import datasets


eval_logger = logging.getLogger(__name__)


try:
    import antlr4
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


# CoT version with step-by-step reasoning prompt
def doc_to_text_cot(doc: dict) -> str:
    return (
        "Problem:" + "\n" + 
        doc["problem"] + "\n\n" + 
        "Let's think step by step to solve this problem.\n\n" +
        "Solution:"
    )


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples_cot() -> list[dict]:
    """CoT few-shot examples with explicit step-by-step reasoning"""
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
            "solution": "Let's think step by step to solve this problem.\n\nStep 1: Identify the constraints from the square roots.\nFor $\\sqrt{x-2}$ to be defined, we need $x-2 \\ge 0$, which means $x \\ge 2$.\nFor $\\sqrt{5-x}$ to be defined, we need $5-x \\ge 0$, which means $x \\le 5$.\n\nStep 2: Consider the denominator constraint.\nSince we have $\\sqrt{5-x}$ in the denominator, it cannot be zero.\nSo we need $5-x > 0$, which means $x < 5$.\n\nStep 3: Combine all constraints.\nWe need $x \\ge 2$ AND $x \\le 5$ AND $x < 5$.\nThis gives us $2 \\le x < 5$, or in interval notation: $[2,5)$.\n\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "Let's think step by step to solve this problem.\n\nStep 1: Recall the determinant property for matrix multiplication.\nFor any two square matrices $\\mathbf{A}$ and $\\mathbf{B}$, we have:\n$\\det(\\mathbf{A} \\mathbf{B}) = \\det(\\mathbf{A}) \\cdot \\det(\\mathbf{B})$\n\nStep 2: Apply the property with given values.\nWe are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$.\n\nStep 3: Calculate the result.\n$\\det(\\mathbf{A} \\mathbf{B}) = \\det(\\mathbf{A}) \\cdot \\det(\\mathbf{B}) = 2 \\cdot 12 = 24$\n\nTherefore, $\\det (\\mathbf{A} \\mathbf{B}) = \\boxed{24}$.\nFinal Answer: The final answer is $24$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "Let's think step by step to solve this problem.\n\nStep 1: Calculate the total weight lifted with 20-pound weights.\nTerrell lifts two 20-pound weights 12 times.\nTotal weight = 2 weights × 20 pounds/weight × 12 times = 480 pounds\n\nStep 2: Set up the equation for 15-pound weights.\nLet $n$ be the number of times he needs to lift the 15-pound weights.\nWith two 15-pound weights lifted $n$ times:\nTotal weight = 2 weights × 15 pounds/weight × $n$ times = 30$n$ pounds\n\nStep 3: Set the total weights equal and solve.\nWe want the same total weight, so:\n30$n$ = 480\n\nStep 4: Solve for $n$.\n$n = \\frac{480}{30} = 16$\n\nTherefore, Terrell must lift the 15-pound weights $\\boxed{16}$ times.\nFinal Answer: The final answer is $16$. I hope it is correct.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "Let's think step by step to solve this problem.\n\nStep 1: Analyze the system for consistency.\nFor the system to have a solution where both $x$ and $y$ are nonzero, the equations must be dependent (since we have 2 equations in 2 unknowns, and we want infinitely many solutions).\n\nStep 2: Make the coefficients of one variable the same.\nLet's manipulate the first equation to match the second equation's pattern.\nMultiply the first equation by $-\\frac{3}{2}$:\n$-\\frac{3}{2}(6x - 4y) = -\\frac{3}{2}a$\n$-9x + 6y = -\\frac{3}{2}a$\n\nStep 3: Rearrange to match the second equation.\nRearranging: $6y - 9x = -\\frac{3}{2}a$\n\nStep 4: Compare with the second equation.\nWe have:\n- From step 3: $6y - 9x = -\\frac{3}{2}a$\n- Given: $6y - 9x = b$\n\nStep 5: Set them equal and solve.\nSince both expressions equal $6y - 9x$:\n$-\\frac{3}{2}a = b$\n\nTherefore: $\\frac{a}{b} = -\\frac{2}{3}$\n\nThe answer is $\\boxed{-\\frac{2}{3}}$.\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)

    if is_equiv(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    # math_verify
    _mvres = verify(
        gold=parse(doc["solution"]),
        target=parse(candidates),
    )
    mathval = 1 if _mvres else 0

    res = {
        "exact_match": retval,
        "math_verify": mathval,
    }
    return res


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
