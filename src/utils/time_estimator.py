import json
import os
import re
from io import BytesIO
from typing import Dict, List

import syllables
from g2p_en import G2p
from mutagen.mp3 import MP3
from openai import OpenAI

from .constants import openai_api_key
from .fs_wrapper import FastSpeechWrapper
from .tool import remove_citation, remove_subtitles


class LengthEstimator:
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "fastspeech":
            self.client = FastSpeechWrapper(batch_size=8)
        elif self.mode == "openai":
            self.client = OpenAI(api_key=openai_api_key)

    def query_time(self, content: List[str], mode=None) -> List[float]:
        if mode is not None and mode != self.mode:
            if self.mode == "fastspeech":
                self.client = FastSpeechWrapper(batch_size=2)
            elif self.mode == "openai":
                self.client = OpenAI(api_key=openai_api_key)

        if isinstance(content, str):
            content = [content]
        clean_content = [remove_citation(c)[0] for c in content]
        clean_content = [remove_subtitles(c) for c in clean_content]
        if self.mode == "words":
            length = [LengthEstimator.count_words(c) for c in clean_content]
        elif self.mode == "syllables":
            length = [LengthEstimator.count_syllables(c) for c in clean_content]
        elif self.mode == "phonemes":
            length = [LengthEstimator.count_phonemes(c) for c in clean_content]
        elif self.mode == "fastspeech":
            length = self.client.query_time(clean_content)
            length = [l * 1.11 - 7 if l > 100 else l for l in length]  # fit openai speed
        elif self.mode == "openai":
            length = []
            for c in clean_content:
                response = self.client.audio.speech.create(model="tts-1", voice="echo", input=c[:4096])

                audio_bytes = BytesIO(response.content)

                length.append(MP3(audio_bytes).info.length)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        if len(length) == 1:
            return length[0]
        return length

    @staticmethod
    def count_words(text):
        """
        Count the number of words in a text string.

        Args:
            text (str): The input text to count words from

        Returns:
            int: Number of words in the text

        Features:
        - Handles multiple spaces/newlines
        - Considers hyphenated words as single words
        - Treats contractions as single words
        - Ignores standalone punctuation
        - Handles multiple languages
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text.strip():
            return 0

        # Replace multiple spaces/newlines with single space
        text = " ".join(text.split())

        # Handle special cases
        def is_word(token):
            # Check if token contains at least one letter or number
            return any(c.isalnum() for c in token)

        # Split on spaces and filter out non-words
        words = [word for word in text.split() if is_word(word)]

        return len(words)

    @staticmethod
    def count_syllables(text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text.strip():
            return 0

        # Replace multiple spaces/newlines with single space
        text = " ".join(text.split())

        # Handle special cases
        def is_word(token):
            # Check if token contains at least one letter or number
            return any(c.isalnum() for c in token)

        # Split on spaces and filter out non-words
        words = [word for word in text.split() if is_word(word)]
        n_count = syllables.estimate(" ".join(words))

        return n_count

    @staticmethod
    def count_phonemes(text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text.strip():
            return 0

        # Replace multiple spaces/newlines with single space
        text = " ".join(text.split())

        # Handle special cases
        def is_word(token):
            # Check if token contains at least one letter or number
            return any(c.isalnum() for c in token)

        # Split on spaces and filter out non-words
        words = [word for word in text.split() if is_word(word)]

        g2p = G2p()
        phonemes = [x for x in g2p(" ".join(words)) if x != " "]
        n_count = len(phonemes)

        return n_count


if __name__ == "__main__":
    estimator = LengthEstimator("fastspeech")
    # estimator = LengthEstimator("openai")
    content = [
        """Thank you very much. So I think that if you want to invest in tires, you should invest in tires. I think that there is income inequality happening in the United States. There is education inequality. There is a planet which is slowly becoming uninhabitable if you look at the Flint water crisis. If you look at droughts that happen in California all the time and if you want to help, these are real problems that exist that we need to help people who are currently not having all of their basic human rights fulfilled. These are things that the government should be investing money in and should probably be investing more money in because we see them being problems in our society that are hurting people. What I’m going to do in this speech is I’m going to continue talking about these criteria, continue talking about why we're not meeting basic needs and why also the market itself is probably solving this problem already. Before that, two points of rebuttal to what we just heard from Project Debater. So firstly, we heard that this is technology that would end up benefiting society but we're not sure we haven't yet heard evidence that shows us why it would benefit all of society, perhaps some parts of society, maybe upper middle class or upper class citizens could benefit from these inspiring research, could benefit from the technological innovations. But most of society, people who are currently in the United States have resource scarcity, people who are hungry, people who do not have access to good education, aren't really helped by this. So we think it is like that, a government subsidy should go to something that helps everyone particularly weaker classes in society. Second point is this idea of an exploding industry which creates jobs and international cooperation. So firstly, we've heard evidence that this already exists, right? We've heard evidence that companies are investing in this as is. And secondly, we think that international cooperation or the specific things have alternatives. We can cooperate over other types of economic trade deals. We can cooperate in other ways with different countries. It's not necessary to very specifically fund space 98  exploration to get these benefits. So as we remember, there are two criteria that I believe the government needs to meet before subsidizing something. It being a basic human need, we don't see space exploration meeting that and B, that this is something that can't otherwise exist, right? So we've already heard from Project Debater how huge this industry is, right? How much investment there's already going on in the private sector and we think this is because there's lots of curiosity especially among wealthy people who maybe want to get to space for personal use or who want to build a colony on Mars and then rent out the rooms there. We know that Elon Musk is doing this already. We know that other people are doing it and we think they're spending money and willing to spend even more money because of the competition between them. So Project Debater should know better than all of us how competitions often bear extremely impressive fruit, right? We think that when wealthy philanthropist or people who are willing to fund research on their own race each other to be the first to achieve new heights in terms of space exploration, that brings us to great achievements already and we think that the private market is doing this well enough already. Considering that we already have movement in that direction, again we see Elon Musk's company, we see all of these companies working already. We think that it's not that the government money won't help out if it were to be given, we just think it doesn't meet the criteria in comparison to other things, right? So given the fact that the market already has a lot of money invested in this, already has movement in those research directions, and given the fact that we still don't think this is a good enough plan to prioritize over other basic needs that the government should be providing people. We think that at the end of the day, given the fact that there are also alternatives to getting all of these benefits of international cooperation, it simply doesn't justify specifically the government allocating its funds for this purpose when it should be allocating them towards other needs of other people."""
    ]
    print(estimator.query_time(content))
