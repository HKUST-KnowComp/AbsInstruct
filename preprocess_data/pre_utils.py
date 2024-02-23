import re
import time
import random
import openai
import asyncio
from string import punctuation

punctuation = set(punctuation)


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        rate_limit_retry_num = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # retry on all exceptions and errors
            except Exception as e:
                print(f"Try count: {rate_limit_retry_num}, Error: {e}")
                # Increment retries
                rate_limit_retry_num += 1

                # Check if max retries has been reached
                if rate_limit_retry_num > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                print(f"Failure, sleep {delay} secs")
                time.sleep(delay)

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry_with_exponential_backoff
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def my_title(string):
    token_list = string.split()
    token_list = [token.capitalize() for token in token_list]
    token_list = "".join(token_list)
    return token_list


# Step1: explain the original word
# Step2: explain the concept word
# Step3: No. The meaning of \"office supplies\" does not encompass \"faxes of lists.\"
noun_exemplar_list = [
    {"event": "<faxes of lists> become available",
     "instance": "faxes of lists",
     "concept": "office supplies",
     "label": 0,
     "instance_exp": "\"Faxes of lists\" refers to a specific type of document that is sent or received "
                     "using a fax machine.",
     "concept_exp": "\"Office supplies\" generally refers to various items that are used in an office setting.",
     "conclusion": "No. The meaning of \"office supplies\" does not encompass \"faxes of lists.\""},

    {"event": "PeopleX's <country> has become known",
     "instance": "country",
     "concept": "cuisine",
     "label": 0,
     "instance_exp": "\"Country\" refers to a nation with its own government.",
     "concept_exp": "\"Cuisine\" refers to styles of cooking of a particular country or region.",
     "conclusion": "No. The meaning of \"cuisine\" does not encompass \"country.\""},
    {"event": "the table bears <details seen>",
     "instance": "details seen",
     "concept": "recognizable aspects",
     "label": 1,
     "instance_exp": "\"Details seen\" implies specific information or elements that "
                     "can be observed or noticed.",
     "concept_exp": "\"Recognizable aspects\" suggests that there are specific features or characteristics that "
                    "can be identified or distinguished.",
     "conclusion": "Yes, the meaning of \"recognizable aspects\" does encompass \"details seen.\""},

    {"event": "ford has given it another <try>",
     "instance": "try",
     "concept": "attempt",
     "label": 1,
     "instance_exp": "\"Try\" refers to an effort or endeavor to accomplish something.",
     "concept_exp": "\"Attempt\" typically refers to an act of trying to achieve something.",
     "conclusion": "Yes."},

    {"event": "a patient should be given <aspirin>",
     "instance": "aspirin",
     "concept": "antiplatelet agent",
     "label": 1,
     "instance_exp": "\"Aspirin\" is a specific noun referring to a medication commonly used "
                     "for pain relief, fever reduction, and anti-inflammatory purposes.",
     "concept_exp": "\"Antiplatelet agent\" is a broader term that encompasses medications that "
                    "prevent blood clotting by inhibiting platelet function.",
     "conclusion": "Yes. The meaning of \"antiplatelet agent\" encompasses \"aspirin\"."},

    {"event": "the <communal experience of theater> will remain important to consumers",
     "instance": "communal experience of theater",
     "concept": "emotions",
     "label": 0,
     "instance_exp": "The phrase \"communal experience of theater\" refers to the shared experience of "
                     "watching a theatrical performance in a group setting.",
     "concept_exp": "\"Emotions\" refers to the various feelings and reactions that individuals experience.",
     "conclusion": "No. While emotions can be a part of the communal experience of theater, "
                   "it does not encompass the entire meaning of that phrase."},

    {"event": "the majority <of stars> were formed in associations",
     "instance": "of stars",
     "concept": "supernovae",
     "label": 0,
     "instance_exp": "The phrase \"of stars\" refers to a group or collection of stars.",
     "concept_exp": "The word \"supernovae\" refers to exploding stars at the end of their "
                    "life cycle.",
     "conclusion": "No. The meaning of \"supernovae\" does not encompass \"of stars.\""},

    {"event": "PeopleX are in <thermal equilibrium with other>",
     "instance": "thermal equilibrium with other",
     "concept": "equilibrium state",
     "label": 1,
     "instance_exp": "\"Thermal equilibrium with other\" refers to a specific type of "
                     "equilibrium where the temperature of a system is equal to the temperature of "
                     "its surroundings.",
     "concept_exp": "The word \"equilibrium state\" refers to a state of balance or "
                    "stability.",
     "conclusion": "Yes. The meaning of \"equilibrium state\" encompasses \"thermal "
                   "equilibrium with other.\""},

    {"event": "mr. altman gives savannah the <murky exoticism>",
     "instance": "murky exoticism",
     "concept": "feeling",
     "label": 0,
     "instance_exp": "\"Murky exoticism\" is a specific phrase that refers to "
                     "a particular type of mysterious or obscure foreignness.",
     "concept_exp": "The meaning of \"feeling\" is a general term that encompasses a wide "
                    "range of emotions or sensations.",
     "conclusion": "No, the meaning of \"feeling\" does not encompass \"murky exoticism.\""},

    {"event": "the <flames> were so hot",
     "instance": "flames",
     "concept": "heat source",
     "label": 1,
     "instance_exp": "The noun \"flames\" refers to the visible, gaseous part of a fire that emits heat and light.",
     "concept_exp": "The noun \"heat source\" refers to any object or substance that generates heat.",
     "conclusion": "Yes. The meaning of \"heat source\" encompasses \"flames.\""},
]

verb_exemplar_list = [
    {"event": "PersonX was <gunned> down by firing simultaneously",
     "instance": "gunned",
     "concept": "shoot",
     "label": 1,
     "instance_exp": "The verb \"gunned\" in the sentence means to shoot someone with a gun.",
     "concept_exp": "The verb \"shoot\" also means to fire a gun.",
     "conclusion": "Yes, the meaning of \"shoot\" encompasses \"gunned.\""},

    {"event": "PeopleX <offered> PersonX the opportunity",
     "instance": "offered",
     "concept": "supply",
     "label": 0,
     "instance_exp": "The verb \"offered\" means to present or give something to someone for their acceptance or "
                     "rejection.",
     "concept_exp": "The verb \"supply\" means to provide or make available something that is needed or "
                    "wanted.",
     "conclusion": "No. The meaning of \"supply\" does not encompass \"offered.\""},

    {"event": "it <cost> PeopleX PeopleX's jobs",
     "instance": "cost",
     "concept": "ask",
     "label": 0,
     "instance_exp": "The verb \"cost\" refers to the act of requiring the payment of an amount of money.",
     "concept_exp": "Meanwhile, the verb \"ask\" refers to the act of making a request or an inquiry.",
     "conclusion": "No, the meaning of \"ask\" does not encompass \"cost.\""},

    {"event": "PeopleX resumed <dating>",
     "instance": "dating",
     "concept": "go forth",
     "label": 0,
     "instance_exp": "The verb \"date\" means to go out with someone romantically or socially.",
     "concept_exp": "The verb \"go forth\" means to proceed, continue, or move forward.",
     "conclusion": "No, the meaning of \"go forth\" does not encompass \"dating.\""},

    {"event": "PersonY <start> believing",
     "instance": "start",
     "concept": "commence",
     "label": 1,
     "instance_exp": "The verb \"start\" means to begin or initiate something.",
     "concept_exp": "The verb \"commence\" also means to begin or start something.",
     "conclusion": "Yes, the meaning of \"commence\" encompasses \"start.\""},

    {"event": "PersonX was <commissioned> an ensign in navy",
     "instance": "commissioned",
     "concept": "give",
     "label": 0,
     "instance_exp": "\"Commissioned\" means to grant or bestow a specific rank or position, "
                     "often in a formal or official capacity.",
     "concept_exp": "On the other hand, the verb \"give\" generally means to transfer possession or control of "
                    "something to someone else.",
     "conclusion": "No."},

    {"event": "ideas <work> for mothers",
     "instance": "work",
     "concept": "operate",
     "label": 1,
     "instance_exp": "The verb \"work\" refers to have the result or effect that you want.",
     "concept_exp": "Meantime, the verb \"operate\" typically refers to perform as expected when applied.",
     "conclusion": "Yes, the meaning of \"operate\" encompasses \"work.\""},

    {"event": "PersonX <respect> everyone",
     "instance": "respect",
     "concept": "regard",
     "label": 1,
     "instance_exp": "The verb \"respect\" means to have admiration, esteem, or high regard for someone.",
     "concept_exp": "The verb \"regard\" means to consider or think about someone or something in a particular way.",
     "conclusion": "Yes. The meaning of \"regard\" encompasses \"respect.\""},

    {"event": "orchards of cultivar were <established>",
     "instance": "established",
     "concept": "open up",
     "label": 0,
     "instance_exp": "The verb \"established\" means to set up or create something, such as an organization, "
                     "system, or institution.",
     "concept_exp": "The verb \"open up\" means to make something accessible or available, "
                    "to reveal or expose something.",
     "conclusion": "No, the meaning of \"open up\" does not encompass \"established.\""},

    {"event": "ms. anderson <replied>",
     "instance": "replied",
     "concept": "say",
     "label": 1,
     "instance_exp": "\"Replied\" specifically refers to responding or answering to something that has been said or "
                     "asked.",
     "concept_exp": "The verb \"say\" generally refers to expressing something through speech.",
     "conclusion": "Yes, the meaning of \"say\" encompasses \"replied.\""},
]

event_exemplar_list = [
    {"event": "<things were going wrong>",
     "instance": "<things were going wrong>",
     "concept": "challenges arising",
     "label": 1,
     "instance_exp": "The sentence \"<things were going wrong>\" suggests that there were "
                     "problems or difficulties occurring.",
     "concept_exp": "On the other hand, the abstract description \"challenges arising\" implies "
                    "that there are new or developing difficulties.",
     "conclusion": "Yes."},

    {"event": "<it had stopped beating>",
     "instance": "<it had stopped beating>",
     "concept": "heart cessation",
     "label": 0,
     "instance_exp": "The sentence \"<it had stopped beating>\" describes the act of something ceasing to beat.",
     "concept_exp": "The abstract description \"heart cessation\" refers to the heart ceasing to beat.",
     "conclusion": "No."},

    {"event": "<the company had not yet seen the complaint>",
     "instance": "<the company had not yet seen the complaint>",
     "concept": "unfamiliarity with the complaint",
     "label": 1,
     "instance_exp": "The sentence \"<the company had not yet seen the complaint>\" suggests that the company has "
                     "not yet received or become aware of the complaint.",
     "concept_exp": "On the other hand, the abstract description \"unfamiliarity with "
                    "the complaint\" implies a lack of knowledge or awareness about the complaint.",
     "conclusion": "Yes."},

    {"event": "<all is keep working hard>",
     "instance": "<all is keep working hard>",
     "concept": "diligence",
     "label": 1,
     "instance_exp": "The sentence \"<all is keep working hard>\" suggests that everyone is putting in effort and "
                     "working diligently.",
     "concept_exp": "The abstract description \"diligence\" refers to the quality of being persistent "
                    "and hardworking.",
     "conclusion": "Yes"},

    {"event": "<the price of returns is big risks>",
     "instance": "<the price of returns is big risks>",
     "concept": "dividends",
     "label": 0,
     "instance_exp": "The sentence \"<the price of returns is big risks>\" seems to convey the idea that there are "
                     "significant risks associated with obtaining high returns or profits.",
     "concept_exp": "On the other hand, the abstract description \"dividends\" refers to a portion of a company's "
                    "profits that is distributed to its shareholders.",
     "conclusion": "No."},

    {"event": "<an suv is built using a body-on-frame platform>",
     "instance": "<an suv is built using a body-on-frame platform>",
     "concept": "body-on-frame design",
     "label": 0,
     "instance_exp": "The sentence \"<an SUV is built using a body-on-frame platform>\" describes a happened "
                     "action of constructing an SUV, specifically using a body-on-frame platform.",
     "concept_exp": "The abstract description \"body-on-frame design\" refers to the construction method, "
                    "where a vehicle has a separate body and frame.",
     "conclusion": "No. The meaning of the sentence \"<an SUV is built using a body-on-frame platform>\" is not "
                   "encompassed by the abstract description \"body-on-frame design.\""},
    {"event": "<nothing seemed impossible>",
     "instance": "<nothing seemed impossible>",
     "concept": "boundless potential",
     "label": 1,
     "instance_exp": "The sentence \"<nothing seemed impossible>\" conveys a sense of limitless possibilities and "
                     "a belief that there are no limitations or obstacles.",
     "concept_exp": "The abstract description \"boundless potential\" also conveys a similar idea of unlimited "
                    "possibilities and the absence of restrictions or boundaries.",
     "conclusion": "Yes."},

    {"event": "<PersonY's resignation would become effective>",
     "instance": "<PersonY's resignation would become effective>",
     "concept": "transition",
     "label": 1,
     "instance_exp": "The sentence \"<PersonY's resignation would become effective>\" refers to the action of "
                     "PersonY's resignation becoming official or taking effect.",
     "concept_exp": "The abstract description \"transition\" can refer to the process of changing from one state or "
                    "condition to another.",
     "conclusion": "Yes."},

    {"event": "<PersonX felt great>",
     "instance": "<PersonX felt great>",
     "concept": "positive mindset",
     "label": 0,
     "instance_exp": "The sentence \"<PersonX felt great>\" conveys that PersonX is experiencing a positive "
                     "emotional state or is in a state of well-being.",
     "concept_exp": "On the other hand, the abstract description \"positive mindset\" implies a general attitude or "
                    "mindset that is optimistic, happy, and focused on the positive aspects of life.",
     "conclusion": "No"},
    {"event": "<PersonX do n't use the word>",
     "instance": "<PersonX do n't use the word>",
     "concept": "refraining from speaking",
     "label": 0,
     "instance_exp": "The sentence \"<PersonX do n't use the word>\" means that PersonX is being instructed not to "
                     "use a specific word.",
     "concept_exp": "The abstract description \"refraining from speaking\" suggests a general act of "
                    "not speaking, without specifying any specific word.",
     "conclusion": "No."},
]
