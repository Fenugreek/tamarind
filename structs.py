"""
Some useful data structures and functions missing from standard python.
"""

class Trie:
    """
    Implement a prefix tree.
    
    Adapted from https://stackoverflow.com/questions/11015320/how-to-create-a-trie-in-python/11015381#11015381
    """

    def __init__(self, sequences, store=False, _end='_end_'):
        """
        Given a list of lists, construct and store a prefix tree.
        The leaves are 2-tuples of the form (depth, sequence_index).

        store: store the given sequences as self.sequences.
        
        _end: use this as the key for the leaves of the tree.
        """
        self._end = _end
        if store:
            self.sequences = sequences
        
        root = dict()
        for index, sequence in enumerate(sequences):
            current_dict = root
            for depth, token in enumerate(sequence):
                current_dict = current_dict.setdefault(token, {})
            current_dict[_end] = (depth, index)

        self.trie = root
        

    def contains(self, sequence):
        """Returns whether given sequence is in the prefix tree."""
        
        current_dict = self.trie
        for token in sequence:
            if token not in current_dict:
                return False
            current_dict = current_dict[token]
        return self._end in current_dict


    def match(self, sequence):
        """
        Returns number of elements into the given sequence that traverse the
        prefix tree successfully, and
        a) the leaf if that traversal gets to a leaf, or
        b) None if otherwise.
        
        Result will be (<depth>, None|<leaf>), where
        0 <= depth <= len(sequence).
        """
        count, current_dict = 0, self.trie
        for token in sequence:
            if token not in current_dict:
                break
            current_dict = current_dict[token]
            count += 1
        return count, current_dict.get(self._end, None)
    

def nested_count(obj):
    """
    Where <obj> is a nested dict or list or combination thereof,
    return count of total leaf elements.
    """
    count = 0
    stack = [obj]
    while len(stack):
        item = stack.pop()
        if type(item) == list:
            for subitem in item:
                if type(subitem) == list or type(subitem) == dict:
                    stack.append(subitem)
                else:
                    count += 1
        elif hasattr(item, 'values'):
            stack.extend(list(item.values()))
        else:
            raise ValueError(f"Encountered type {type(item)} that's neither list nor dict.")
    return count
