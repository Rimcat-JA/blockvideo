"""Experimental DeepL translation helper for Japanese project content.

Imports:
    ``deepl`` supplies the external translation client.
    ``os`` reads the optional ``DEEPL_KEY`` environment variable.

Module state:
    ``AUTH_KEY`` contains the process-level environment value, and ``client``
    is created once at import time.  This helper is not part of the generation
    pipeline yet; callers should expect network access, provider-dependent
    failures, and the need for a configured DeepL key.
"""

import os
import deepl

# Optional environment-level DeepL credential used by the experimental helper.
AUTH_KEY = os.getenv('DEEPL_KEY', default=None)
# One reusable client for all sentence translations in this process.
client = deepl.DeepLClient(AUTH_KEY)

def translate(contents: list[str], target_lang: str) -> list[str]:
    """Translate each supplied content item through DeepL.

    Args:
        contents: Non-empty sequence of source strings.  Non-string values are
            converted with ``str`` before submission.
        target_lang: DeepL target-language code such as ``"EN-US"``.

    Returns:
        Translation results in input order, excluding provider results that
        are ``None`` or empty.

    Raises:
        TypeError: If ``contents`` is ``None``.
        ValueError: If ``contents`` is empty.
        Exception: Provider/client errors are intentionally allowed to bubble
            up because this experimental helper has no fallback policy.

    Side Effects:
        Performs one network translation request per content item.

    """
    # Exclude edge situations
    if contents is None:
        raise TypeError('Contents should not be None.')
    if len(contents) == 0:
        raise ValueError('Contents should not be empty.')

    # Call translator API to translate for each sentence
    translated_contents: list[str] = []
    for content in contents:
        if type(content) != str:
            content = str(content)

        result = client.translate_text(content, target_lang=target_lang)

        # Append to the translated contents list if the result is not None
        if (result is not None) and (len(result) != 0):
            translated_contents.append(result)

    return translated_contents
