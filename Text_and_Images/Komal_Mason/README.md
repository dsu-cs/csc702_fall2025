In this project, Komal and I worked on using an image generator to create various images alongside various textual descriptions of images. 
We then took this and converted all of these to embeddings. We then measured the cosine similarity between the image embeddings to that of the textual descriptions. 
We used a small variety of text descriptions describing various things. 
We then implemented various realistic images as well as unrealistic images 
(with this, we showed that many textual descriptions likely don't represent these irregular images, but may relate more to realistic images). 
Furthermore, we followed this by showing some examples of our prompts for images and text by comparing these embeddings and determining which class of images 
have more similar descriptions. We found that in general the irregular images were generally dissimilar to the realistic textual descriptions 
whereas the realistic images generally had more similarities to the realistic descriptions. This shows perhaps that these irregular images 
can sometimes be harder to read for machines as they may not have textual captions to relate to text for them. Likewise, it may be difficult because of this 
when you have no actual images of, say, a giant Centaur on the moon raiding alien outposts, to actually create an image of such. Whereas, if you have 
1,000,000,000 images of peoples cats, it may be easier to create images of such images. I believe this is reflected by the realism and quality of our 
realistic vs. unrealistic prompts and the accompanying images.
