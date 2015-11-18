#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import codecs
import re
import logging
import argparse
import traceback
import bz2


logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CleanText:
    MIN_WORD_NUMBER         = 1
    MAX_WORD_NUMBER         = 80

    RE_QUOTE                = ur"(^|\s)&quot;(\s|$)"
    RE_HYPHEN               = ur"\s-\s"
    RE_SLASH                = ur"\s\/\s"
    RE_APOSTROPHE           = ur"(^|\s)&apos;(\s|$)"
    RE_ISBN                 = ur"[Ii][Ss][Bb][Nn]"
    RE_DOT_ASTERIX          = ur"\s\.\s\*\s"
    RE_ASTERIX_ONLY         = ur"(?:^|\s)\*(?:\s|$)"
    RE_MATHEMATICAL_FORMULA = ur"[+−]+.*[+−]+.*"
    RE_NUMBERS              = ur"[0-9]+(?:\.[0-9]+)?"
    RE_TABLE_ENTRY          = ur"((^|\s)!\s!(\s|$).*){1,}"
    RE_PUNCTUATION          = ur"[.,;:!?¿¡]+"
    RE_TRANSFORMED_IE       = ur"i__PUNC__e__PUNC__"
    RE_TRANSFORMED_EG       = ur"e__PUNC__g__PUNC__"
    RE_TRANSFORMED_US       = ur"u__PUNC__s__PUNC__"
    RE_TRANSFORMED_ST       = ur"st__PUNC__"
    RE_TRANSFORMED_NO       = ur"no__PUNC__"
    RE_TRANSFORMED_DR       = ur"dr__PUNC__"
    RE_TRANSFORMED_VS       = ur"vs__PUNC__"
    RE_TRANSFORMED_DC       = ur"d__PUNC__c__PUNC__"
    RE_TRANSFORMED_BA       = ur"b__PUNC__a__PUNC__"
    RE_TRANSFORMED_PM       = ur"p(?:__PUNC__)?m__PUNC__"


    RE_TEMPERATURE              = ur"__NUM__\s°\s[fc]"
    RE_TRANSFORMED_APOSTROPHE   = ur"(^|\s)&apos__PUNC__"
    RE_TRANSFORMED_AMPERSAND    = ur"(^|\s)&amp__PUNC__"

    NUMBER_SUBSTITUTION         = ur"__NUM__"
    PUNCTUATION_SUBSTITUTION    = ur"__PUNC__"
    APOSTROPHE_SUBSTITUTION     = ur" &apos;"
    AMPERSAND_SUBSTITUTION      = ur" &amp;"
    HYPHEN_SUBSTITUTION         = ur" __HYPHEN__ "
    TEMPERATURE_SUBSTITUTION    = ur"__TEMP__"


    def __init__(self, lang):
        self.language = lang


    def is_balanced_parenthesis(self, text):
        #if len(text)%2!=0:
        #    return False
        opening=set('(')
        match=set([ ('(',')') ])
        stack=[]
        for char in text:
            if not char in ['(', ')']: # allow for any text
                continue
            if char in opening:
                stack.append(char)
            else:
                if len(stack)==0:
                    return False
                lastOpen=stack.pop()
                if (lastOpen, char) not in match:
                    return False
        return len(stack)==0


    def filter_noise(self, input_sentence):
        _ACCENTED_CHARS = ur"áéíóúàèìòùäëïöüâêîôûãẽĩõũÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛÃẼĨÕŨñÑ"
        _GERMAN_AND_OTHER_CHARS   = ur"ßæēōāıīł"
        _CURRENCY_AND_OTHER_SYMBOLS = ur"\%\$£€¥"
        _VALID_CHARS = _ACCENTED_CHARS + _GERMAN_AND_OTHER_CHARS + _CURRENCY_AND_OTHER_SYMBOLS

        _EG             = ur"\s?e\.g\.\s?"
        _IE             = ur"\s?i\.e\.\s?"
        _US             = ur"\s?u\.s\.\s?"
        _ST             = ur"\s?st\.\s?"
        _NO             = ur"\s?no\.\s?"
        _VS             = ur"\s?vs\.\s?"
        _DR             = ur"\s?dr\.\s?"
        _DC             = ur"\s?d\.c\.\s?"
        _BA             = ur"\s?b\.a\.\s?"

        _TEMP           = ur"\s?__TEMP__\s?"
        _HYPHEN         = ur"\s?__HYPHEN__\s?"
        _PUNC           = ur"\s?__PUNC__\s?"
        _NUM            = ur"\s?-?__NUM__\s?-?"
        _AMP_HTML       = ur"\s?&amp;\s?"
        _APOS_HTML      = ur"\s?&apos;\s?"
        _QUOT_HTML      = ur"\s?&quot;\s?"
        _OPEN_PARENS    = ur"\s?\(\s?"
        _CLOSE_PARENS   = ur"\s?\)\s?"
        _TOK_HYPHEN     = ur"\s?@-@\s?"

        _EXCEPTIONS =   _EG         + "|" + _IE             + "|" + _US             + "|" + _ST         + "|" + \
                        _VS         + "|" + _DC             + "|" + _BA             + "|" + _NO         + "|" + \
                        _DR         + "|" +                                                                     \
                        _TEMP       + "|" + _HYPHEN         + "|" +                                             \
                        _PUNC       + "|" + _NUM            + "|" + _AMP_HTML       + "|" + _APOS_HTML  + "|" + \
                        _QUOT_HTML  + "|" + _OPEN_PARENS    + "|" + _CLOSE_PARENS   + "|" + _TOK_HYPHEN

        # final regexp to filter out noise
        RE_VALID_TERMS = ur"("+_EXCEPTIONS+")|[^A-Z"+_VALID_CHARS+"a-z\s]+|([A-Za-z"+_VALID_CHARS+"]+\s)"
        
        m = [group for group in re.findall(RE_VALID_TERMS, input_sentence, re.UNICODE)
              if (group[0].strip()!="" or group[1].strip()!="")]

        ret = ""
        for group in m:
            if group[0].strip()!="":
                ret += group[0]
            if group[1].strip()!="":
                ret += group[1]
        
        # remove repetitive whitespace
        ret = re.sub(ur"(\s)\1+", ur"\1", ret)

        return ret


    def process_line(self, line_to_process):
        line = line_to_process.strip()

        # remove lines that do not have open and close parenthesis balanced
        if not self.is_balanced_parenthesis( line ):
            pass
            #yield None
        else:
            # remove apostrophes in case of english wikipedia
            if self.language == "en":
                line = re.sub(self.RE_APOSTROPHE, " ", line)

            # remove any quotes and any apostrophes not followed by "s"
            line = re.sub(self.RE_QUOTE, " ", line)

            # split if a line contains multiple sentences separated by "end of sentence one . * Start of sentence two . * Sentence three ."
            line = re.sub(self.RE_DOT_ASTERIX, " .\n", line)
            number_of_lines = len(line.split("\n"))

            # if line contains less than MIN_WORD_NUMBER of more than MAX_WORD_NUMBER, skip it
            for line_new1 in line.split("\n"):
                line_new1 += "\n"
                line_new1 = re.sub(self.RE_QUOTE, " ", line_new1)

                if self.language == "en":
                    line_new1 = re.sub(self.RE_APOSTROPHE, " ", line_new1)

                #line_new1 = re.sub(self.RE_ASTERIX_ONLY, "\n", line_new1)
                line_new1 = re.sub(self.RE_ASTERIX_ONLY, " ", line_new1)

                for line_new2 in line_new1.split("\n"):
                    #line_new2 = re.sub(self.RE_ASTERIX_ONLY, "\n", line_new2)
                    line_new2 = re.sub(self.RE_ASTERIX_ONLY, " ", line_new2)
                    for line_new in line_new2.split("\n"):
                        #line_new = line_new2.strip()+"\n"
                        #line_new += "\n"
                        N_WORDS = len(line_new.split())

                        # if line contains a table entry, skip it
                        if re.search(self.RE_TABLE_ENTRY, line_new) != None:
                            continue;

                        # too much or too few words in line
                        if (N_WORDS < self.MIN_WORD_NUMBER or N_WORDS > self.MAX_WORD_NUMBER):
                            continue;

                        # too many numbers (likely to be a weird table line or similar)
                        # more than 50% of words being numbers in that case
                        if len(re.findall(self.RE_NUMBERS, line_new)) / float(N_WORDS) > 0.5:
                            continue;

                        # too many punctuation marks and numbers combined
                        if ( len(re.findall(self.RE_PUNCTUATION, line_new)) +   \
                             len(re.findall(self.RE_HYPHEN, line_new)) +        \
                             len(re.findall(self.RE_QUOTE, line_new)) +         \
                             len(re.findall(self.RE_APOSTROPHE, line_new)) +    \
                             len(re.findall(ur"\(|\)", line_new)) +             \
                             len(re.findall(self.RE_NUMBERS, line_new)) ) / float(N_WORDS) > 0.30:
                            continue;

                        # too many • symbols (likely to be part of a weird table again)
                        res = re.findall(ur"•", line_new, re.UNICODE)
                        len_res = len(res)
                        if len_res / float(N_WORDS) > 0.25:
                            continue;


                        # remove lines that contain ISBN book entries
                        if re.search(self.RE_ISBN, line_new) != None:
                            continue;

                        # remove lines that contain mathematical formula
                        if re.search(self.RE_MATHEMATICAL_FORMULA, line_new) != None:
                            continue;

                        # substitute numbers per one symbol for all numbers
                        line_new = re.sub(self.RE_NUMBERS, self.NUMBER_SUBSTITUTION, line_new)

                        # substitute slash per string "or"
                        if self.language == "en":
                            line_new = re.sub(self.RE_SLASH, ur" or ", line_new)
                        if self.language == "de":
                            line_new = re.sub(self.RE_SLASH, ur" oder ", line_new)
                        if self.language == "fr":
                            line_new = re.sub(self.RE_SLASH, ur" ou ", line_new)


                        # substitute hyphens per one symbol
                        line_new = re.sub(self.RE_HYPHEN, self.HYPHEN_SUBSTITUTION, line_new)

                        # substitute punctuation per one symbol for all punctuation marks
                        line_new = re.sub(self.RE_PUNCTUATION, self.PUNCTUATION_SUBSTITUTION, line_new)

                        # apostrophes followed by s will have been converted. write them back to what they were
                        line_new = re.sub(self.RE_TRANSFORMED_APOSTROPHE, self.APOSTROPHE_SUBSTITUTION, line_new)

                        # temperature in the format "__NUM__ ° c" should be transformed
                        line_new = re.sub(self.RE_TEMPERATURE, self.TEMPERATURE_SUBSTITUTION, line_new)

                        # the same happens with expressions "i.e." and "e.g." and the string " u.s.",
                        # write them back to what they were
                        line_new = re.sub(self.RE_TRANSFORMED_IE, ur"i.e.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_EG, ur"e.g.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_US, ur"u.s.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_ST, ur"st.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_NO, ur"no.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_DR, ur"dr.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_VS, ur"vs.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_DC, ur"d.c.", line_new)
                        line_new = re.sub(self.RE_TRANSFORMED_BA, ur"b.a.", line_new)

                        # the same happens with ampersands. write them back to what they were
                        line_new = re.sub(self.RE_TRANSFORMED_AMPERSAND, self.AMPERSAND_SUBSTITUTION, line_new)

                        logger.debug(line_new)

                        line_new = self.filter_noise(line_new)

                        # if after filtering out noise line contains empty parameters, remove them
                        line_new = re.sub(ur"\(\s\)", ur" ", line_new)

                        logger.debug(line_new)

                        #fh_out.write(line_new.strip()+" "+unicode(len(re.findall(RE_NUMBERS, line_new)) / N_WORDS)+"\n")
                        #self._fh_out.write(line_new.strip()+"\n")
                        if line_new.strip() == "":
                            continue
                        if line_new.strip().startswith(ur"__PUNC__"):
                            continue

                        yield line_new.strip()+u"\n"



if __name__ == "__main__":
    # process command line
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true', required=False, default=False)
    p.add_argument('--input-file', required=False, type=str, default=None,
                help="File containing corpus extracted from Wikipedia, one sentence per line. Defaults to stdin.")
    p.add_argument("--output-file", required=False, type=str, default=None,
                help="File with outputted cleaned corpus. Defaults to stdout.")
    p.add_argument("--language", required=True, type=str, choices=("en", "de", "fr"),
                help="Language of the files.")
    args = p.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.input_file is None:
        logger.info("Reading input from stdin...")
        fh = codecs.getreader('utf8')(sys.stdin)
    else:
        if args.input_file.lower().endswith('bz2'):
            logger.info('Reading compressed input file %s' % args.input_file)
            fh = bz2.BZ2File(args.input_file, 'r')
        else:
            fh = codecs.open(args.input_file, 'r', 'utf-8')
    if args.output_file is None:
        logger.info("Writing output to stdout...")
        fh_out = codecs.getwriter('utf8')(sys.stdout)
    else:
        fh_out = codecs.open(args.output_file, 'w', 'utf-8')

    c = CleanText(args.language)

    try:
        counter = 0
        counter_split = 0
        for line in fh:
            if not isinstance(line, unicode): # necessary for processing bzipped data (ascii-encoded by default)
                #logger.info('Encoding line as unicode: %s' % line)
                line = str.decode(line, 'utf8')

            # process line
            output_lines = c.process_line( line )
            if not output_lines is None:
                output = u"\n".join(output_lines)
                number_of_lines = len( output.split("\n") ) 
                # remove empty lines
                output = re.sub(ur"(\n)\1+", ur"\1", output)

                fh_out.write( output )

                counter += 1
                counter_split += number_of_lines
                if counter % 1000000 == 0:
                    logger.info( "Processed %d lines (%d including splitted lines) ..." % (counter, counter_split) )
    except:
        traceback.print_exc()

    logger.info( "Done!" )
    fh.close()
    fh_out.close()

