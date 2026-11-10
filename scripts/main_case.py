import gc

from reuse_methods import *

#logging.basicConfig(level=logging.DEBUG)

par = Parser("licenses_description.yml")

# 6 Corpus, 5 Image DS, 2 Music DS, 1 3D DS, 1 Video DS, 10 Models

wiki_text = Work('Wikipedia', 'data', 'raw', 'CC-BY-SA-4.0') # Corpus, https://en.wikipedia.org/wiki/Wikipedia:Copyrights
stack_exchange_text = Work('StackExchange', 'data', 'raw', 'CC-BY-SA-4.0') # Corpus, https://stackexchange.com/
free_law_text = Work('FreeLaw', 'data', 'raw', 'CC-BY-ND-4.0') # Corpus, https://free.law/
arxiv_text = Work('arXiv', 'data', 'raw', 'CC-BY-NC-SA-4.0') # Corpus, https://info.arxiv.org/help/license/index.html
pubmed_text = Work('PubMed', 'data', 'raw', 'CC-BY-NC-ND-4.0') # Corpus, https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/
deep_sequoia_text = Work('Deep-sequoia', 'data', 'raw', 'LGPL-LR') # Corpus, http://deep-sequoia.inria.fr/

midjourney_img = Work('Midjourney_gen', 'data', 'raw', 'CC-BY-NC-4.0') # Image, https://docs.midjourney.com/docs/terms-of-service
flickr_img = Work('Flickr', 'data', 'raw', 'CC-BY-NC-SA-4.0') # Image, https://www.flickr.com/creativecommons/
stocksnap_img = Work('StockSnap', 'data', 'raw', 'CC0-1.0') # Image, https://stocksnap.io/license
wikimedia_img = Work('Wikimedia', 'data', 'raw', 'CC-BY-SA-4.0') # Image, https://commons.wikimedia.org/wiki/Main_Page
open_clipart_img = Work('OpenClipart', 'data', 'raw', 'CC0-1.0') # Image, https://openclipart.org/share

ccmixter_music = Work('ccMixter', 'data', 'raw', 'CC-BY-NC-4.0') # Music, https://ccmixter.org/terms 
jamendo_music = Work('Jamendo', 'data', 'raw', 'CC-BY-NC-ND-4.0') # Music, https://www.jamendo.com/legal/creative-commons

thingverse_3d = Work('Thingverse', 'data', 'raw', 'CC-BY-NC-SA-4.0') # 3D Model, https://www.thingiverse.com/thing:4850246
vimeo_video = Work('Vimeo', 'data', 'raw', 'CC-BY-NC-ND-4.0') # Video, https://vimeo.com/creativecommons

baize_model = Work('Baize', 'model', 'raw', 'GPL-3.0') # Chatbot, https://github.com/project-baize/baize-chatbot
stable_diffusion_mode = Work('StableDiffusion', 'model', 'raw', 'CreativeML-OpenRAIL-M') # Text2Image, https://huggingface.co/runwayml/stable-diffusion-v1-5
whisper_model = Work('Whisper', 'model', 'raw', 'MIT') # Voice2Text, https://github.com/openai/whisper
maskformer_model = Work('MaskFormer', 'model', 'raw', 'CC-BY-NC-4.0') # Image Segmentation, https://github.com/facebookresearch/MaskFormer/tree/da3e60d85fdeedcb31476b5edd7d328826ce56cc
detr_model = Work('DETR', 'model', 'raw', 'Apache-2.0') #Image Segmentation https://github.com/facebookresearch/detr
xclip_model = Work('X-Clip', 'model', 'raw', 'MIT') # Video2Text, https://huggingface.co/microsoft/xclip-base-patch32
i2vgen_model = Work('I2VGen-XL', 'model', 'raw', 'CC-BY-NC-ND-4.0') # Image2Video, https://huggingface.co/damo-vilab/MS-Image2Video
bigtranslate_model = Work('BigTranslate', 'model', 'raw', 'GPL-3.0') # Text Translation, https://huggingface.co/James-WYang/BigTranslate
nllb_model = Work('NLLB-200', 'model', 'raw', 'Apache-2.0') # Text Translation, https://huggingface.co/facebook/nllb-200-distilled-600M
bert_model = Work('BERT', 'model', 'raw', 'Apache-2.0') # Text, https://huggingface.co/bert-base-uncased
bloom_model = Work('BLOOM', 'model', 'raw', 'BigScience-BLOOM-RAIL-1.0') # Text Generation, https://huggingface.co/bigscience/bloom
llama2_model = Work('Llama2', 'model', 'raw', 'Llama2') # Text Generation, https://huggingface.co/meta-llama/Llama-2-7b
opt_175b_model = Work('OPT-175B', 'model', 'raw', 'OPT-175B') # Text Generation, https://huggingface.co/facebook/opt-175b

# Coverage-extension components (LicBench L067–L074)
agpl_tool_model = Work('AGPL-Tool', 'model', 'raw', 'AGPL-3.0')
lgpl_lib_model = Work('LGPL-Lib', 'model', 'raw', 'LGPL-3.0')
bsd_model = Work('BSD-Model', 'model', 'raw', 'BSD-3-Clause')
openrail_pp_model = Work('OpenRAIL++-Model', 'model', 'raw', 'OpenRAIL++')
open_corpus_text = Work('OpenCorpus', 'data', 'raw', 'CC-BY-4.0')

# Register License for above works
works = [ob for ob in gc.get_objects() if isinstance(ob, Work)]
par.register_license(works)
