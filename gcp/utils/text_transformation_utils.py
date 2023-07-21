import re


class TextTransformations:
    def __init__(self, text):
        self.text = text

    def lowercase(self):
        """Converts to lowercase"""
        return [line.lower() for line in self.text]

    def decontractions(self):
        """Performs decontractions in the doc"""
        new_text = []
        for phrase in self.text:
            phrase = re.sub(r"won't", "will not", phrase)
            phrase = re.sub(r"can\'t", "can not", phrase)
            phrase = re.sub(r"couldn\'t", "could not", phrase)
            phrase = re.sub(r"shouldn\'t", "should not", phrase)
            phrase = re.sub(r"wouldn\'t", "would not", phrase)
            # general
            phrase = re.sub(r"n\'t", " not", phrase)
            phrase = re.sub(r"\'re", " are", phrase)
            phrase = re.sub(r"\'s", " is", phrase)
            phrase = re.sub(r"\'d", " would", phrase)
            phrase = re.sub(r"\'ll", " will", phrase)
            phrase = re.sub(r"\'t", " not", phrase)
            phrase = re.sub(r"\'ve", " have", phrase)
            phrase = re.sub(r"\'m", " am", phrase)
            phrase = re.sub(r"\*+", "abuse", phrase)
            new_text.append(phrase)

        return new_text

    def rem_punctuations(self):
        """Removes punctuations"""
        punctuations = """
                            !()-[]{};:'"\,<>/?@#$%^&*~
                        """
        new_text = []
        for line in self.text:
            for char in line:
                if char in punctuations:
                    line = line.replace(char, "")
            new_text.append(' '.join(line.split()))
        return new_text

    def rem_numbers(self):
        """Removes numbers and irrelevant text like xxxx*"""
        new_text = []
        for line in self.text:
            temp = re.sub(r'x*', '', line)
            new_text.append(re.sub(r'\d', '', temp))
        return new_text

    def words_filter(self):
        """Removes words less than 2 characters except no and ct"""
        new_text = []
        for line in self.text:
            temp = line.split()
            temp2 = [word for word in temp if len(word) > 2 or word == 'no' or word == 'ct']
            new_text.append(' '.join(temp2))
        return new_text

    def multiple_fullstops(self):
        """ Removes multiple full stops from the text"""
        return [re.sub(r'\.\.+', '.', line) for line in self.text]

    def fullstops(self):
        return [re.sub('\.', ' .', line) for line in self.text]

    def multiple_spaces(self):
        return [' '.join(line.split()) for line in self.text]

    def separting_startg_words(self):
        new_text = []
        for line in self.text:
            temp = []
            words = line.split()
            for i in words:
                if not i.startswith('.'):
                    temp.append(i)
                else:
                    w = i.replace('.', '. ')
                    temp.append(w)
            new_text.append(' '.join(temp))
        return new_text

    def rem_apostrophes(self):
        return [re.sub("'", '', line) for line in self.text]
