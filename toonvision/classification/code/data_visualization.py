# %% Imports
import matplotlib.pyplot as plt
from data_processing import RAW_DIR, UNSORTED_DIR
from img_utils import extract_objects_from_xml
from collections import Counter
from glob import glob
from statistics import mean


# %% Global variables
ALL_LABELS = [
    "cog_bb_flunky",
    "cog_bb_pencilpusher",
    "cog_bb_yesman",
    "cog_bb_micromanager",
    "cog_bb_downsizer",
    "cog_bb_headhunter",
    "cog_bb_corporateraider",
    "cog_bb_thebigcheese",
    "cog_lb_bottomfeeder",
    "cog_lb_bloodsucker",
    "cog_lb_doubletalker",
    "cog_lb_ambulancechaser",
    "cog_lb_backstabber",
    "cog_lb_spindoctor",
    "cog_lb_legaleagle",
    "cog_lb_bigwig",
    "cog_cb_shortchange",
    "cog_cb_pennypincher",
    "cog_cb_tightwad",
    "cog_cb_beancounter",
    "cog_cb_numbercruncher",
    "cog_cb_moneybags",
    "cog_cb_loanshark",
    "cog_cb_robberbaron",
    "cog_sb_coldcaller",
    "cog_sb_telemarketer",
    "cog_sb_namedropper",
    "cog_sb_gladhander",
    "cog_sb_movershaker",
    "cog_sb_twoface",
    "cog_sb_themingler",
    "cog_sb_mrhollywood",
    "toon_bear",
    "toon_cat",
    "toon_crocodile",
    "toon_deer",
    "toon_dog",
    "toon_duck",
    "toon_horse",
    "toon_monkey",
    "toon_mouse",
    "toon_pig",
    "toon_rabbit"
]
ANIMALS = [
    "bear",
    "cat",
    "crocodile",
    "deer",
    "dog",
    "duck",
    "horse",
    "monkey",
    "mouse",
    "pig",
    "rabbit",
]
BINARY = ["cog", "toon"]
SUITS_LONG = ["bossbot", "lawbot", "cashbot", "sellbot"]
SUITS_SHORT = ["bb", "lb", "cb", "sb"]


# %% Define functions
def plot_suits_as_histogram() -> None:
    # all_cogs = glob(f"{UNSORTED_DIR}/cog/*.png")
    # num_cogs = len(all_cogs)

    all_suits = [glob(f"{UNSORTED_DIR}/cog/cog_{suit}*.png") for suit in SUITS_SHORT]
    num_suits = [len(suit) for suit in all_suits]
    # all_suits_dict = dict(zip(SUITS, num_suits))
    # print(num_cogs, all_suits_dict)

    # Histogram of all_suits_dict
    plt.bar(SUITS_SHORT, num_suits)
    plt.title("Number of labels per suit")
    plt.xlabel("Suit")
    plt.show()


def plot_toons_as_histogram() -> None:
    # all_toons = glob(f"{UNSORTED_DIR}/toon/toon_*.png")
    # num_toons = len(all_toons)

    all_animals = [
        glob(f"{UNSORTED_DIR}/toon/toon_{animal}*.png") for animal in ANIMALS
    ]
    num_animals = [len(animal) for animal in all_animals]
    # all_animals_dict = dict(zip(ANIMALS, num_animals))
    # print(num_toons, all_animals_dict)

    # Histogram of all_animals_dict
    plt.bar(ANIMALS, num_animals)
    plt.title("Number of labels per animal")
    plt.xlabel("Animal")
    plt.show()


def plot_xml_data() -> None:
    def get_binary_label(obj_name: str) -> str:
        obj_type = obj_name.split("_")[0]
        if obj_type in BINARY:
            return obj_type
        return "unknown"

    def get_suit_label(obj_name: str) -> str:
        if "cog_bb" in obj_name:
            return "bossbot"
        elif "cog_lb" in obj_name:
            return "lawbot"
        elif "cog_cb" in obj_name:
            return "cashbot"
        elif "cog_sb" in obj_name:
            return "sellbot"
        else:
            return "unknown"

    def get_animal_label(obj_name: str) -> str:
        return obj_name.split("_")[1]

    def update_counters(obj_name: str) -> None:
        count_all.update([obj_name])
        count_binary.update([get_binary_label(obj_name)])
        if obj_name.startswith("cog"):
            count_suit.update([get_suit_label(obj_name)])
        if obj_name.startswith("toon"):
            count_animal.update([get_animal_label(obj_name)])

    def set_counters_to_zero() -> None:
        count_all.update({key: 0 for key in ALL_LABELS})
        count_binary.update({key: 0 for key in BINARY})
        count_suit.update({key: 0 for key in SUITS_LONG})
        count_animal.update({key: 0 for key in ANIMALS})

    all_xml = glob(f"{RAW_DIR}/screenshots/*/*.xml")

    # Create counters
    count_all = Counter()     # All object names (32 + 11 = 43 classes)
    count_binary = Counter()  # Cog or Toon (2 classes)
    count_suit = Counter()    # Bossbot, Lawbot, Cashbot, Sellbot (4 classes)
    count_animal = Counter()  # Toon animals (11 classes)
    set_counters_to_zero()

    for xml in all_xml:
        objs = extract_objects_from_xml(xml)
        for obj_name, _, _, _, _ in objs:
            update_counters(obj_name)

    # * Sort by highest value
    # sorted_all_labels = dict(count_all.most_common())
    # * Sort by key values
    # count_all_sorted = {key: count_all[key] for key in sorted(count_all.keys())}

    # Counter obj, title, desired_counter, axis,
    counts_and_titles = [
        (count_all, "Number of objects per label", 20),
        (count_binary, "Number of objects per binary label", 640),
        (count_suit, "Number of objects per suit label", 160),
        (count_animal, "Number of objects per animal label", 58),
    ]
    fig, ax = plt.subplots(4, 1, figsize=(5, 15), gridspec_kw={"height_ratios": [4, 1, 1, 1]})
    fig.suptitle("Number of labels in unprocessed screenshots")
    for idx, (count_dict, title, desired_count) in enumerate(counts_and_titles):
        hbars = ax[idx].barh(y=list(count_dict.keys()), width=count_dict.values())
        ax[idx].invert_yaxis()
        ax[idx].axvline(x=mean(count_dict.values()), color="red", alpha=0.5)
        ax[idx].axvline(x=desired_count, color="green")
        ax[idx].set_title(title)
        ax[idx].bar_label(hbars, count_dict.values())
    # ax[0].set_height(20)
    plt.show()


# %% Plot data
plot_suits_as_histogram()
plot_toons_as_histogram()
plot_xml_data()
