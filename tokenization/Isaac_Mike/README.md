Imports data from Gutenburg, we use Dracula.

Byte Pair Encoding implementation

We do some clean up on the text before byte pairing by removing stop words.

The learning function returns a list of the merges and another function create the valid token vocab from that. We use the token vocab to tokenzise inputs after that.

