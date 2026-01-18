import re


async def clean_latex_content(text: str) -> str:
    """Clean LaTeX while keeping math & structure."""

    # Remove comments
    text = re.sub(r'%.*', '', text)

    # Remove preamble
    text = re.sub(r'\\documentclass[\s\S]*?\\begin{document}', '', text)

    # Remove end
    text = re.sub(r'\\end{document}', '', text)

    # Remove badly converted latex artifacts
    text = re.sub(r'\\(?:url|href|cite|label|ref|footnote)\{.*?\}', '', text)

    # Remove uncompiled environments
    text = re.sub(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}', '', text)
    text = re.sub(r'\\includegraphics[\s\S]*?}', '', text)

    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
