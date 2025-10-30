import os
import sys

import numpy as np
import torch
import yaml

fastspeech_path = os.path.join(os.path.dirname(__file__), "..", "..", "dependencies")
sys.path.append(fastspeech_path)

from fastspeech2.synthesize import preprocess_english
from fastspeech2.utils.model import get_model_2, get_vocoder
from fastspeech2.utils.tools import synth_samples, synth_samples_for_length, to_device

root = f"{fastspeech_path}/fastspeech2/"


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


class FastSpeechWrapper:
    def __init__(self, batch_size=8):
        preprocess_config = yaml.load(open(f"{root}/config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(f"{root}/config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(f"{root}/config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader)
        self.configs = (preprocess_config, model_config, train_config)

        pitch_control, energy_control, duration_control = 1.0, 1.0, 1.0
        restore_step = 900000
        self.control_values = pitch_control, energy_control, duration_control

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model_2(restore_step, self.configs, self.device, train=False)
        self.vocoder = get_vocoder(model_config, self.device)
        self.batch_size = batch_size

    def process_text(self, texts):
        batchs = []
        for i, text in enumerate(texts):
            ids = f"Part-{i}"
            raw_text = text
            speakers = 0
            post_text = np.array(preprocess_english(text, self.configs[0]))
            text_len = len(post_text)
            batchs.append((ids, raw_text, speakers, post_text, text_len))
        return batchs

    def query_time(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batchs = self.process_text(texts)
        lengths = self.synthesize(batchs)
        return lengths

    def synthesize(self, batchs):
        preprocess_config, model_config, train_config = self.configs
        pitch_control, energy_control, duration_control = self.control_values

        lengths = []
        batchs = [batchs[i : i + self.batch_size] for i in range(0, len(batchs), self.batch_size)]
        for batch in batchs:
            ids = [d[0] for d in batch]
            speakers = np.array([d[2] for d in batch])
            texts = [d[3] for d in batch]
            raw_texts = [d[1] for d in batch]
            text_lens = np.array([d[4] for d in batch])
            texts = pad_1D(texts)
            data = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))

            data = to_device(data, self.device)
            with torch.no_grad():
                # Forward
                output = self.model(
                    *(data[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control
                )
                length = synth_samples_for_length(
                    data,
                    output,
                    self.vocoder,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                )
                lengths.extend(length)
        return lengths


if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        """Friends, colleagues, fellow citizens, we're here today to discuss a critical issue: how we ensure everyone has access to the healthcare they need, especially when resources are stretched thin.  We believe the government has a vital role to play in this, and we affirm the resolution that the government should step in to ration care when absolutely necessary.
Now, I know “rationing” can be a loaded word. It might conjure up images of arbitrary decisions, of someone deciding whose life is more valuable. But that's not what we're advocating for.  Our stance is about fairness, about creating a just and equitable system for allocating limited resources when we simply don't have enough for everyone.  This isn't some theoretical debate; it's about real-life situations where tough choices are unavoidable. It's about ensuring the long-term health and viability of our healthcare system for all of us.
Let's be clear about what we mean by rationing.  We're talking about situations where there simply aren't enough resources to go around. In those instances, the government should step in with clear, ethical guidelines to prioritize who gets what, making sure the most people benefit and everyone is treated fairly.
Our first point is this: rationing ensures *equitable access* to essential healthcare. Think about it: without government oversight, access is often determined by how much money you have.  This creates a two-tiered system where the wealthy get the best care, while those struggling financially are left behind.  An article in the *Journal of Medical Ethics*, "Rationing health care and the need for credible scarcity: why Americans can't say no," argues that a market-based system inherently favors the rich, leaving vulnerable populations with limited options. Government rationing, while challenging, levels the playing field, putting medical need ahead of ability to pay.  It's about making sure everyone has a fair shot at the care they need, regardless of their bank account.
Second, transparent government frameworks build *public trust*.  When decisions are made in the open, with clear criteria, people are more likely to trust the system.  The *American Journal of Public Health* article, "Restoring Trust In Our Nation's Public Health System," directly links transparency to increased public trust.  Another study, "Transparency during public health emergencies: from rhetoric to reality," published in *Public Health*, emphasizes how transparency promotes fairness, accountability, and trust, especially during crises.  Open communication and clear rules for resource allocation show a commitment to fairness and allow for public scrutiny. This is a stark contrast to market-driven systems where decisions are often made behind closed doors.
Finally, rationing promotes *responsible resource management*.  The COVID-19 pandemic showed us just how fragile our healthcare systems can be when resources are scarce.  Even wealthy countries had to consider rationing.  "The realities of rationing in health care," published in *Health Affairs*, highlights this very point.  By implementing transparent and ethical rationing frameworks, governments can make the most of limited resources, preventing system collapse and ensuring long-term viability.  This proactive approach protects the system for everyone, prioritizing the well-being of society as a whole.
So, while rationing presents complex ethical questions, it becomes a necessary tool when resources are simply too scarce. Government intervention, guided by transparency and a commitment to equity, ensures that limited resources are used responsibly, maximizing benefit and preserving the healthcare system for all. We believe that a just and equitable system, while difficult, is the most ethical approach in the face of scarcity.""",
        """Friends, neighbors, let's talk healthcare.  We've all been there – anxiously waiting for a doctor's appointment, or facing a hefty medical bill.  We all want a system that's fair, efficient, and puts people first.
My opponents express concern about government involvement in healthcare rationing. They paint a picture of a discriminatory system.  Now, while there have been instances of bias in the past – and we shouldn't shy away from that – those instances actually highlight why we *need* strong, transparent government oversight.  Think of it like this:  we have traffic laws to prevent accidents, not because driving is inherently bad.  Similarly, government frameworks ensure fairness and are open to public scrutiny, minimizing the risk of bias.  My opponents cite a 1999 study on racial disparities in liver transplants.  But consider this: that study predates many of the equity-focused reforms we've implemented since then. We're not defending past mistakes; we're building a better future.
They also claim government rationing stifles innovation.  Let's look at the COVID-19 pandemic.  Remember the rapid development of new treatments, telehealth, and resource allocation strategies?  That happened during a period of intense government intervention.  The article, “Ten things to consider when implementing rationing guidelines during a pandemic,” highlights this very point.  The pandemic proved government involvement can *spur* innovation.  Comparing our proposal to the Soviet Union's healthcare system is just a distraction.  Our system has democratic checks and balances, transparency, and public accountability – things the Soviet system lacked.
Now, let's talk alternatives.  My opponents suggest market-based approaches and private charity.  But think about it: market-based systems prioritize profit.  “Access to Healthcare and Disparities in Access” shows how market forces can leave vulnerable populations behind.  Private charity, while well-intentioned, simply can't handle nationwide healthcare allocation.  It's inconsistent and often driven by donor preferences, not actual need.  As “Rationing is not the only alternative” points out, relying solely on these creates a fragmented, inequitable system.  And the lack of transparency in private systems, highlighted in “PDF,” makes it hard to ensure fairness.  Our government-led system operates in the open, accountable to all of us.
Furthermore, government rationing, done right, maximizes health outcomes.  When resources are scarce, it prioritizes those most likely to benefit, as “Resource Utilization in the Emergency Department—The Duty of Stewardship” explains.  This ensures resources achieve the greatest possible good, unlike systems that prioritize ability to pay.  “The realities of rationing in health care” supports this.  Ethical frameworks, overseen by the government, guide these difficult decisions.  “Informing the Gestalt: An Ethical Framework for Allocating Scarce Resources” and the New Zealand government's “Ethics, Equity and Resource Allocation” provide examples of such frameworks.
Finally, government oversight minimizes the influence of private interests and mitigates market failures.  “Why healthcare market needs government intervention to improve access and efficiency” and “Governance in Health” emphasize the importance of government in balancing public and private interests, ensuring healthcare is driven by societal needs, not just profit.
In conclusion, a well-designed, transparent system overseen by the government is the most ethical and effective way to allocate scarce healthcare resources.  It ensures equitable access, responsible resource management, and adherence to ethical principles.  The alternatives simply don't offer the same level of fairness, accountability, or effectiveness.""",
        """Friends, colleagues, let's talk about something vital to us all: healthcare.  Think about your last trip to the doctor, or a loved one needing urgent care.  Now imagine not being able to afford it.  That's the reality for many when resources are scarce, and that's why we're here today.  The question is: how do we ensure *everyone* gets the care they need when there simply isn't enough to go around?
We believe the answer lies in carefully designed government oversight.  Think of it like traffic lights – they're there to ensure everyone gets through the intersection safely and fairly, not just the fastest cars.  Government frameworks, with clear rules and open communication, bring that same fairness to healthcare.  Research, like the 2022 study by Leider et al. published in *[Journal Name]*, shows that transparency builds public trust, especially during tough times.  Remember the early days of the pandemic?  Clear government guidance was crucial then, unlike the often confusing messages from private systems.
Now, our opponents talk about market-based solutions and charity.  But even in well-off countries like Switzerland, as the 2023 King's Fund report shows, not everyone gets equal access.  And charity, while kind, is like patching potholes – it helps a few, but it doesn't fix the whole road.  It can't guarantee everyone gets the care they deserve.  Our opponents also claim government slows innovation.  But look at the pandemic!  The rapid development of vaccines and telehealth happened *because* of government funding and coordination.
So, when healthcare is limited, we must prioritize fairness.  Government rationing, with clear rules and public input, isn't perfect, but it's the most just way to ensure everyone, regardless of their income, has a fair chance.  It's about ensuring everyone has access to the care they need, not just those who can afford it.  We urge you to support this vital measure.""",
        """Thank you very much. So I think that if you want to invest in tires, you should invest in tires. I think that there is income inequality happening in the United States. There is education inequality. There is a planet which is slowly becoming uninhabitable if you look at the Flint water crisis. If you look at droughts that happen in California all the time and if you want to help, these are real problems that exist that we need to help people who are currently not having all of their basic human rights fulfilled. These are things that the government should be investing money in and should probably be investing more money in because we see them being problems in our society that are hurting people. What I’m going to do in this speech is I’m going to continue talking about these criteria, continue talking about why we're not meeting basic needs and why also the market itself is probably solving this problem already. Before that, two points of rebuttal to what we just heard from Project Debater. So firstly, we heard that this is technology that would end up benefiting society but we're not sure we haven't yet heard evidence that shows us why it would benefit all of society, perhaps some parts of society, maybe upper middle class or upper class citizens could benefit from these inspiring research, could benefit from the technological innovations. But most of society, people who are currently in the United States have resource scarcity, people who are hungry, people who do not have access to good education, aren't really helped by this. So we think it is like that, a government subsidy should go to something that helps everyone particularly weaker classes in society. Second point is this idea of an exploding industry which creates jobs and international cooperation. So firstly, we've heard evidence that this already exists, right? We've heard evidence that companies are investing in this as is. And secondly, we think that international cooperation or the specific things have alternatives. We can cooperate over other types of economic trade deals. We can cooperate in other ways with different countries. It's not necessary to very specifically fund space 98  exploration to get these benefits. So as we remember, there are two criteria that I believe the government needs to meet before subsidizing something. It being a basic human need, we don't see space exploration meeting that and B, that this is something that can't otherwise exist, right? So we've already heard from Project Debater how huge this industry is, right? How much investment there's already going on in the private sector and we think this is because there's lots of curiosity especially among wealthy people who maybe want to get to space for personal use or who want to build a colony on Mars and then rent out the rooms there. We know that Elon Musk is doing this already. We know that other people are doing it and we think they're spending money and willing to spend even more money because of the competition between them. So Project Debater should know better than all of us how competitions often bear extremely impressive fruit, right? We think that when wealthy philanthropist or people who are willing to fund research on their own race each other to be the first to achieve new heights in terms of space exploration, that brings us to great achievements already and we think that the private market is doing this well enough already. Considering that we already have movement in that direction, again we see Elon Musk's company, we see all of these companies working already. We think that it's not that the government money won't help out if it were to be given, we just think it doesn't meet the criteria in comparison to other things, right? So given the fact that the market already has a lot of money invested in this, already has movement in those research directions, and given the fact that we still don't think this is a good enough plan to prioritize over other basic needs that the government should be providing people. We think that at the end of the day, given the fact that there are also alternatives to getting all of these benefits of international cooperation, it simply doesn't justify specifically the government allocating its funds for this purpose when it should be allocating them towards other needs of other people.""",
    ]

    fs = FastSpeechWrapper(batch_size=2)
    lengths = fs.query_time(texts)
    print(lengths)
