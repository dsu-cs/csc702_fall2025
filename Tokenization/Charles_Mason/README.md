For part of this project, a BPE english tokenizer is created. from this, there are two variations. One is the baseline using modern english books. The other uses some modern english books, but also a lot of old english books. The version with both types that is more specified for old timey texts. These are then both applied to test data and they measure success by analyzing average tokens per line. Each model performs on both a generic and old timey test. The scores are reported. Interesting things are found, these will be discussed later (also noted in the code). Specifically, vocab size looked at for how it affects the performance for both variations. This analyzes both how vocab size can affect tokenization as well as how specifying a type of input for training for tokenizers. I also added some charts and metrics for comparison in this document.

The datasets used are from project gutenberg. 

Results: 

    These results are certainly interesting.

    They show us numerous things.

    To begin, an expected observation is that the average tokens created when the vocab size for all of these show a sharp decrease with increase in vocab size. This directly proves the hypothesized negative correlation between average tokens used in a sentence and vocab size (bigger vocab size = fewer tokens used in sentences). That being said, the larger vocab sizes are slower.

    Another observation was the fact that the generic english scores for the baseline were ALWAYS better than the old timey version. This includes the notable observation that eventually the average tokens for generic english plateau. While this occurs, the old timey one still (very slowly) decreases average tokens used. Furthermore, the gap between how worse it was seemed to lessen as vocab size increased. This is likely because as the vocab got bigger, it just went on to add more generic english words instead of relatively having a higher percentage of tokens being related to old timey english. 

    As for the old timey scores, as hoped, the old timey version performed better always. Also notable with this is that the old timey version continued to decrease in average token length while the generic one stalled. This likely proved that the different count of books was not ideal. Similarly, the old timey model generally improved over the generic model more as the vocab size increased. So, initially, the gap was mostly quite close, but as vocab size increased, it is likely old timey tokens were added instead of generic english tokens, making it more efficient at working with these sentences.

    Another interesting observation is that at 10k and below for a vocab size, the performance for avg tokens for general and old timey tests were about the same (meaning there probably wasn't that much of a learned difference). This dramatically changed after 10k for the old time variation, implying that with a larger vocab size, it may be preferable to use variations that have specializations for what you need.

    In sum, it seemed that with larger vocabs especially, having some specific variations may generally benefit more than hurt, but this may also be due to more general information being in that language. Another way of looking at these results that may help is in terms of average tokens between the generic score and old timey score for the variations. The specialized version had a better score especially at larger vocab sizes.



Another part of this project analyzed various tokenization techniques with BPE and Shakespears works. 
For this, the focus was on how different tokenizers split text, how vocabulary grows, and what trade-offs exist between efficiency and linguistic accuracy.

We implement and compare four tokenizers:
1. Whitespace — simple split on spaces.  
2. Regex — rule-based tokenizer that preserves contractions (e.g., *’tis*, *ne’er*, *o’er*) and hyphenated forms (*self-love*).  
3. Character — every character is a token.  
4. BPE-lite — a tiny Byte-Pair Encoding (BPE) demo that learns subword units (e.g., *th*, *ing*, *fore*).


 Features
Tokenizer Comparison 
  - Tokens per 1,000 characters  
  - Vocabulary size  
  - Average token length  
  - Apostrophe handling  

-

Subword / BPE Details**  
  - First 20 learned merges (e.g., *th*, *he*, *in*, *er*).  
  - Example: tokenization of *therefore* at 0, 50, and 300 merges.  

- Error Analysis 
  - Side-by-side tokenizations of tricky Shakespeare forms (*’Tis, Ne’er, O’er, self-love*).  
 


 Overall, the big focus on this project was the various things you can do with tokenizers when adjusting how they function and how you set them up and whatnot. This can be seen through our various results. 