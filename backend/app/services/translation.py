"""
The origin version of the repository is only for Japanese speakers. However, there are also needs from non-Japanese speakers. To solve this issue, we developed a simple script to translate the Japanese contents to other languages for convenience.

You may be careful to this script as it is still an experimental function.
"""

import deepl
import os

AUTH_KEY = os.getenv('DEEPL_KEY', default=None)
client = deepl.DeepLClient(AUTH_KEY)

def translate(contents: list[str], target_lang: str) -> list[str]:
    """
    Translate the Japanese contents to other languages.
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