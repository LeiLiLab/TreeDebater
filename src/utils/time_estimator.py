import math
import threading
from typing import List, Sequence, Union, overload


from .speech_duration import estimate_speech_seconds
from .tool import remove_citation, remove_subtitles


TTS_INPUT_LIMIT = 4096
_G2P = None
_G2P_LOCK = threading.Lock()


def _audio_chunks(text):
    """Cover all cleaned text within the provider limit, preferring word boundaries."""
    while text:
        end = min(len(text), TTS_INPUT_LIMIT)
        if end < len(text):
            boundary = text.rfind(' ', 0, end)
            if boundary >= 0:
                end = boundary + 1
        yield text[:end]
        text = text[end:]


class LengthEstimator:
    def __init__(self, mode, *, audio_duration=None, tts_backend="openai"):
        self.tts_backend = tts_backend
        self.mode = mode
        self.audio_duration = audio_duration
        if self.mode == "fastspeech":
            from .fs_wrapper import get_shared_wrapper
            self.client = get_shared_wrapper(batch_size=8)
        elif self.mode == "openai" and not callable(audio_duration):
            raise ValueError('OpenAI duration estimation requires an injected audio_duration '
                             'callable using the configured TTS client and transport')

    @overload
    def query_time(self, content: str, mode=None) -> float: ...

    @overload
    def query_time(self, content: Sequence[str], mode=None) -> List[float]: ...

    def query_time(self, content: Union[str, Sequence[str]], mode=None) -> Union[float, List[float]]:
        """Strings return a scalar; batches always return a list, including zero/one items."""
        if mode is not None and mode != self.mode:
            return LengthEstimator(mode, audio_duration=self.audio_duration, tts_backend=self.tts_backend).query_time(content)
        scalar = isinstance(content, str)
        if not scalar and not isinstance(content, Sequence):
            raise TypeError('content must be text or a sequence of text')
        inputs = [content] if scalar else content
        if any(not isinstance(c, str) for c in inputs):
            raise TypeError('every content item must be text')
        clean_content = [remove_subtitles(remove_citation(c)[0]) for c in inputs]
        if self.mode == "words":
            length = [LengthEstimator.count_words(c) for c in clean_content]
        elif self.mode == "syllables":
            length = [LengthEstimator.count_syllables(c) for c in clean_content]
        elif self.mode == "phonemes":
            length = [LengthEstimator.count_phonemes(c) for c in clean_content]
        elif self.mode == "time":
            length = [estimate_speech_seconds(c) for c in clean_content]
        elif self.mode == "fastspeech":
            length = self.client.query_time(clean_content) if clean_content else []
            if self.tts_backend == "openai":
                length = [l * 1.11 - 7 if l > 100 else l for l in length]  # fit openai speed
        elif self.mode == "openai":
            length = []
            for text in clean_content:
                total = 0.
                for chunk in _audio_chunks(text):
                    seconds = float(self.audio_duration(chunk))
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError('Audio duration must be finite and nonnegative')
                    total += seconds
                length.append(total)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        if len(length) != len(inputs):
            raise ValueError('Duration estimator returned the wrong batch size')
        return length[0] if scalar else length

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
        import syllables
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

        global _G2P
        with _G2P_LOCK:
            if _G2P is None:
                from g2p_en import G2p
                _G2P = G2p()
            return sum(phone != ' ' for phone in _G2P(" ".join(words)))


if __name__ == "__main__":
    estimator = LengthEstimator("time")
    # estimator = LengthEstimator("openai")
    content = [
        """Thank you very much. So I think that if you want to invest in tires, you should invest in tires. I think that there is income inequality happening in the United States. There is education inequality. There is a planet which is slowly becoming uninhabitable if you look at the Flint water crisis. If you look at droughts that happen in California all the time and if you want to help, these are real problems that exist that we need to help people who are currently not having all of their basic human rights fulfilled. These are things that the government should be investing money in and should probably be investing more money in because we see them being problems in our society that are hurting people. What I’m going to do in this speech is I’m going to continue talking about these criteria, continue talking about why we're not meeting basic needs and why also the market itself is probably solving this problem already. Before that, two points of rebuttal to what we just heard from Project Debater. So firstly, we heard that this is technology that would end up benefiting society but we're not sure we haven't yet heard evidence that shows us why it would benefit all of society, perhaps some parts of society, maybe upper middle class or upper class citizens could benefit from these inspiring research, could benefit from the technological innovations. But most of society, people who are currently in the United States have resource scarcity, people who are hungry, people who do not have access to good education, aren't really helped by this. So we think it is like that, a government subsidy should go to something that helps everyone particularly weaker classes in society. Second point is this idea of an exploding industry which creates jobs and international cooperation. So firstly, we've heard evidence that this already exists, right? We've heard evidence that companies are investing in this as is. And secondly, we think that international cooperation or the specific things have alternatives. We can cooperate over other types of economic trade deals. We can cooperate in other ways with different countries. It's not necessary to very specifically fund space 98  exploration to get these benefits. So as we remember, there are two criteria that I believe the government needs to meet before subsidizing something. It being a basic human need, we don't see space exploration meeting that and B, that this is something that can't otherwise exist, right? So we've already heard from Project Debater how huge this industry is, right? How much investment there's already going on in the private sector and we think this is because there's lots of curiosity especially among wealthy people who maybe want to get to space for personal use or who want to build a colony on Mars and then rent out the rooms there. We know that Elon Musk is doing this already. We know that other people are doing it and we think they're spending money and willing to spend even more money because of the competition between them. So Project Debater should know better than all of us how competitions often bear extremely impressive fruit, right? We think that when wealthy philanthropist or people who are willing to fund research on their own race each other to be the first to achieve new heights in terms of space exploration, that brings us to great achievements already and we think that the private market is doing this well enough already. Considering that we already have movement in that direction, again we see Elon Musk's company, we see all of these companies working already. We think that it's not that the government money won't help out if it were to be given, we just think it doesn't meet the criteria in comparison to other things, right? So given the fact that the market already has a lot of money invested in this, already has movement in those research directions, and given the fact that we still don't think this is a good enough plan to prioritize over other basic needs that the government should be providing people. We think that at the end of the day, given the fact that there are also alternatives to getting all of these benefits of international cooperation, it simply doesn't justify specifically the government allocating its funds for this purpose when it should be allocating them towards other needs of other people."""
    ]
    print(estimator.query_time(content))
