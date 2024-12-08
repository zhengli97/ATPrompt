from clip import clip
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple
from typing import Dict

import langchain

# import langchain.chains.retrieval_qa.base

langchain.debug = True

class Attr_parser(BaseModel):
    COMMON_ATTRIBUTES: List[str] = Field(
        description="Characteristics or features that are shared among a group of items. These attributes help to categorize and understand how certain items or entities are similar. MUST be one single word.")
parser = PydanticOutputParser(pydantic_object=Attr_parser)
import getpass
import os
API_SECRET_KEY = "ENTER YOUR KEY"
BASE_URL = "https://flag.smarttrot.com/v1/"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# imagenet
# class_name = [
#     "tench", "goldfish", "great white shark", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula, wolf spider", "tick", "centipede", "black grouse, ptarmigan", "ruffed grouse", "prairie grouse", "CD player", "cello", "container ship", "corkscrew", "cornet", "cradle", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "electric fan", "electric guitar", "electric locomotive", "swing", "electrical switch", "syringe", "table lamp", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "sandbar", "beach", "baseball player", "volcano", "scuba diver", "toilet paper", "corn cob", "bolete"
# ]

# caltech101
class_name = [
    "faces", "leopards", "motorbikes", "accordion", "airplanes", "anchor", "ant", "barrel", "bass", "beaver", "binocular", "bonsai", "brain", "brontosaurus", "buddha", "butterfly", "camera", "cannon", "car side", "ceilling fan", "cellphone", "chair", "chandelier", "cougar face", "crab", "crayfish", "crocodile", "crocodile head", "cup", "dalmatian", "dollar bill", "dolphin", "dragonfly", "electric guitar", "elephant", "emu", "euphonium", "ewer", "ferry", "flamingo", "flamingo head", "garfield", "gerenuk", "gramophone", "grand piano", "hawksbill", "headphone", "hedgehog", "helicopter", "ibis", "inline skate", "joshua tree", "kangaroo", "ketch", "lamp", "laptop", "llama", "lobster", "lotus", "mandolin", "mayfly", "menorah", "metronome", "minaret"
]

# oxfordpets
# class_name = [
#     "abyssinian", "american bulldog", "american pit bull terrier", "basset hound", "beagle", "bengal", "birman", "bombay", "boxer", "british shorthair", "chihuahua", "egyptian mau", "english cocker spaniel", "english setter", "german shorthaired", "great pyrenees", "havanese", "japanese chin", "keeshond", "leonberger", "Maine Coon", "miniature pinscher", "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian Blue", "saint bernard", "samoyed", "scottish terrier", "shiba inu", "Siamese", "Sphynx", "staffordshire bull terrier", "wheaten terrier", "yorkshire terrier", "american bulldog", "american pit bull terrier", "basset_hound", "beagle"
# ]

# stanford cars
# class_name = [
#     "2012 Audi TTS Coupe", "2012 Acura TL Sedan", "2007 Dodge Dakota Club Cab", "2012 Hyundai Sonata Hybrid Sedan", "2012 Ford F-450 Super Duty Crew Cab", "1993 Geo Metro Convertible", "2012 Dodge Journey SUV", "2012 Dodge Charger Sedan", "2012 Mitsubishi Lancer Sedan", "2012 Chevrolet Traverse SUV", "2012 Buick Verano Sedan", "2012 Toyota Sequoia SUV", "2007 Hyundai Elantra Sedan", "1997 Dodge Caravan Minivan", "2012 Volvo C30 Hatchback", "1999 Plymouth Neon Coupe", "2007 Chevrolet Malibu Sedan", "2012 Volkswagen Beetle Hatchback", "2007 Chevrolet Corvette Ron Fellows Edition Z06", "2010 Chrysler 300 SRT-8", "2010 BMW M6 Convertible", "2012 GMC Yukon Hybrid SUV", "2012 Nissan Juke Hatchback", "1993 Volvo 240 Sedan", "2012 Suzuki SX4 Sedan", "2010 Dodge Ram Pickup 3500 Crew Cab", "2009 Spyker C8 Coupe", "2012 Land Rover Range Rover SUV", "2012 Hyundai Elantra Touring Hatchback", "2010 Chevrolet Cobalt SS", "2012 Hyundai Veracruz SUV", "2012 Ferrari 458 Italia Coupe", "2012 BMW Z4 Convertible", "2009 Dodge Charger SRT-8", "2012 Fisker Karma Sedan", "2011 Infiniti QX56 SUV", "2012 Audi A5 Coupe", "1991 Volkswagen Golf Hatchback", "2012 GMC Savana Van", "2012 Audi TT RS Coupe", "2012 Rolls-Royce Phantom Sedan", "2012 Porsche Panamera Sedan", "2012 Bentley Continental GT Coupe"
# ]

# flower102
# class_name = [
#     "alpine sea holly", "anthurium", "artichoke", "azalea", "ball moss", "balloon flower", "barbeton daisy", "bearded iris", "bee balm", "bird of paradise", "bishop of llandaff", "black-eyed susan", "blackberry lily", "blanket flower", "bolero deep blue", "bougainvillea", "bromelia", "buttercup", "californian poppy", "camellia", "canna lily", "canterbury bells", "cape flower", "carnation", "cautleya spicata", "clematis", "colt’s foot", "columbine", "common dandelion", "corn poppy", "cyclamen", "daffodil", "desert-rose", "english marigold", "fire lily", "foxglove", "frangipani", "fritillary", "garden phlox", "gaura", "gazania", "geranium", "giant white arum lily", "globe thistle", "globe-flower", "grape hyacinth", "great masterwort", "hard-leaved pocket orchid", "hibiscus", "hippeastrum", "japanese anemone", "king protea", "lenten rose", "lotus", "love in the mist", "magnolia", "mallow", "marigold", "mexican aster", "mexican petunia", "monkshood", "moon orchid", "morning glory", "orange dahlia", "osteospermum", "oxeye daisy", "passion flower", "pelargonium", "peruvian lily", "petunia", "pincushion flower", "pink primrose", "pink-yellow dahlia", "poinsettia", "primula", "prince of wales feathers", "purple coneflower", "red ginger", "rose", "ruby-lipped cattleya", "siam tulip", "silverbush", "snapdragon", "spear thistle", "spring crocus", "stemless gentian", "sunflower", "sweet pea", "sweet william", "sword lily", "thorn apple", "tiger lily", "toad lily", "tree mallow", "tree poppy", "trumpet creeper", "wallflower", "water lily", "watercress", "wild pansy", "windflower", "yellow iris"
# ]

# food101
# class_name = [
#     "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare", "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito", "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake", "ceviche", "cheesecake", "cheese plate", "chicken curry", "chicken quesadilla", "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder", "club sandwich", "crab cakes", "reme brulee", "croque madame", "cup cakes", "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict", "escargots", "falafel", "filet mignon", "fish and chips", "foie gras", "french fries", "french onion soup", "french toast", "fried calamari", "fried rice", "frozen yogurt", "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich", "grilled salmon", "guacamole", "gyoza", "hamburger", "hot and sour soup", "hot dog", "huevos rancheros", "hummus", "ice cream", "lasagna", "lobster bisque", "lobster roll sandwich", "macaroni and cheese", "macarons", "miso soup", "mussels", "nachos", "omelette", "onion rings", "oysters", "pad thai", "paella", "pancakes", "panna cotta", "peking duck", "pho", "pizza", "pork chop", "poutine", "prime rib", "pulled pork sandwich", "ramen", "ravioli", "red velvet cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp and grits", "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak", "strawberry shortcake", "sushi", "tacos", "takoyak", "tiramisu", "tuna tartare", "waffles"
# ]

# fgvc
# class_name = [
#     "707−320", "727−200", "737−200", "737−300", "737−400", "737−500", "737−600", "737−700", "737−800", "737−900", "747−100", "747−200", "747−300", "747−400", "757−200", "757−300", "767−200", "767−300", "767−400", "777−200", "777−300", "A300B4", "A310", "A318", "A319", "A320", "A321", "A330−200", "A330−300", "A340−200", "A340−300", "A340−500", "A340−600", "A380", "ATR−42", "ATR−72", "An−12", "BAE 146−200", "BAE 146−300", "BAE−125"
# ]

# sun397
# class_name = [
#     "abbey", "airplane cabin", "airport terminal", "alley", "amphitheater", "amusement arcade", "amusement park", "anechoic chamber", "apartment building outdoor", "apse indoor", "aquarium", "aqueduct", "archarchive", "arrival gate outdoor", "art gallery", "art school", "art studio", "assembly line", "athletic field outdoor", "atrium publicattic", "auditorium", "auto factory", "badlands", "badminton court indoor", "baggage claim", "bakery shop", "balcony exterior", "balcony interior", "ball pit", "ballroom", "bamboo forest", "banquet hall", "bar", "barn", "barndoor", "baseball field", "basement", "basilica", "basketball court outdoor", "bathroom", "batters box", "bayou", "bazaar indoor", "bazaar outdoor", "beach", "beauty salon", "bedroom", "berth", "biology laboratory", "bistro indoor", "boardwalk", "boat deck", "boathouse", "bookstore", "booth indoor", "botanical garden", "bow window indoor", "bow window outdoor", "bowling alley", "boxing ring", "brewery indoor", "bridge", "building facade", "bullring", "burial chamber", "bus interior", "butchers shop", "butte", "cabin", "cafeteria", "campsite", "campus", "canal natural", "canal urban", "candy store", "canyon", "car interior backseat", "car interior frontseat", "carrousel", "casino", "castle", "catacomb", "cathedral indoor", "cathedral outdoor", "cavern", "cemetery", "chalet", "cheese factory"
# ]

# dtd
# class_name =[
#     "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped", "studded", "swirly", "veined", "waffled", "woven", "wrinkled", "zigzagged"
# ]

# eurosat
# class_name = [
#     "AnnualCrop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial", "Pasture", "Permanent Crop", "Residential", "River", "SeaLake"
# ]

# ucf101
# class_name = [
#     "Apply Eye Makeup", "Apply Lipstick", "Archery", "Baby Crawling", "Balance Beam", "Band Marching", "Baseball Pitch", "Basketball", "Basketball Dunk", "Bench Press", "Biking", "Billiards", "Blow Dry Hair", "Blowing Candles", "Body Weight Squats", "Bowling", "Boxing Punching Bag", "Boxing Speed Bag", "Breast Stroke", "Brushing Teeth", "Clean And Jerk", "Cliff Diving", "Cricket Bowling", "Cricket Shot", "Cutting In Kitchen", "Diving", "Drumming", "Fencing", "Field Hockey Penalty", "Floor Gymnastics", "Frisbee Catch", "Front Crawl", "Golf Swing", "Haircut", "Hammer Throw", "Hammering", "Handstand Pushups", "Handstand Walking", "Head Massage", "High Jump", "Horse Race", "Horse Riding", "Hula Hoop", "Ice Dancing", "Javelin Throw", "Juggling Balls", "Jump Rope", "Jumping Jack", "Kayaking", "Knitting", "Long Jump", "Lunges", "Military Parade", "Mixing", "Mopping Floor", "Nunchucks", "Parallel Bars", "Pizza Tossing", "Playing Cello", "Playing Daf", "Playing Dhol", "Playing Flute", "Playing Guitar", "Playing Piano", "Playing Sitar", "Playing Tabla", "Playing Violin", "Pole Vault", "Pommel Horse", "Pull Ups", "Punch", "Push Ups", "Rafting", "Rock Climbing Indoor", "Rope Climbing", "Rowing", "Salsa Spin", "Shaving Beard", "Shotput", "Skate Boarding", "Skiing", "Skijet", "Sky Diving", "Soccer Juggling", "Soccer Penalty", "Still Rings", "Sumo Wrestling", "Surfing Swing", "Table Tennis Shot"
# ]

prompt1 = ChatPromptTemplate.from_template("Let's think step by step. Please give me a detailed description of the following categories in one sentence (including shape, color, characteristics, habits, functions, etc.): {cls_words}")
prompt2 = ChatPromptTemplate.from_template("Given these visual descriptions: {prompt1}. \n\n Please summarize five general and independent attributes in one noun, \n\n {format_instructions}",partial_variables={"format_instructions": parser.get_format_instructions()},)
chain = prompt1 | model | prompt2 | model | parser

attributes = chain.invoke({"cls_words": class_name}).COMMON_ATTRIBUTES
print(attributes)
