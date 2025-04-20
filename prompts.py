correction_prompt_v1 = """# Correction Task

Given a list of input texts, which may contain grammatical errors, your task is to provide a corrected version of each.

## Input Format

The input will consist of a numbered list of texts.

```
0. text_0
1. text_1
2. ...
```

## Output Format

The output should be a JSON object called "corrected" containing a list of lists consisting of [index, corrected_text]. The index is an int starting from 0, and the corrected_text is either a string, or none. null should be used if the text is already correct and no changes are needed.

```JSON
{{
    "corrected": [
    [0, corrected_text_0,
    [1, corrected_text_1],
    [2, corrected_text_2],
    ...
    ]
}}
```

## Task Notes

- Each output text should be a corrected version of the corresponding input. The corrected text should be grammatically correct and maintain the original meaning as much as possible.
- There must be an exact one-to-one correspondence between the input and output texts, according to the indices. 5 inputs means 5 outputs, even if the number of sentences changes in the process.
  - An input text that is corrected by splitting into more sentences (e.g., replacing a comma with a period) should still be one output that corresponds to the input index.
  - If a single numbered input has multiple parts separated by newlines, make sure to keep them together in the corresponding output.
  - Sentences from separate input indices cannot be merged.
- "Minimal" corrections are considered ideal in this task. There is no need to rewrite for fluency. We are fixing grammatical errors, lexical errors, and orthographic errors.
- The output should not include any additional information or explanations, just the corrected text.

## Examples

### Example 1 Input

```
0. It's difficult answer at the question "what are you going to do in the future?" if the only one who has to know it is in two minds.
1. When I was younger I used to say that I wanted to be a teacher, a saleswoman and even a butcher.. I don't know why.
2. I would like to study Psychology because one day I would open my own psychology office and help people.
3. It's difficult because I'll have to study hard and a lot, but I think that if you like a subject, you'll study it easier.
4. Maybe I'll change my mind, maybe not.
```

### Example 1 Output:

```JSON
{{
  "corrected": [
    [0, "It's difficult to answer the question \"what are you going to do in the future?\" if the only one who has to know it is in two minds."],
    [1, "When I was younger, I used to say that I wanted to be a teacher, a saleswoman and even a butcher. I don't know why."],
    [2, "I would like to study Psychology, because one day I would like to open my own psychology clinic and help people."],
    [3, "It's difficult because I'll have to study hard and a lot, but I think that if you like a subject, you'll study it more easily."],
    [4, null]
  ]
}}
```

### Example 2 Input:

```
0. My favourite sport is volleyball because I love plays with my friends.
1. Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.
2. It is very funny.
3. when I was young I like plays with the ball in the playground and my friend and I played using the soccer goals as a network of volleyball.
```

### Example 2 Output:

```JSON
{{
    "corrected": [
    [0, "My favourite sport is volleyball because I love playing with my friends."],
    [1, "Volleyball is a sport that is played everywhere. When I am on the beach I like playing with my sister in the sand and then we go in the sea."],
    [2, "It is great fun."],
    [3, "When I was young I liked playing with a ball in the playground and my friend and I played using the soccer goals as a volleyball net."]
  ]
}}


## Current Instance

### Current Input

Please correct the following text:

```
{formatted_sentences}
```
"""

edit_extraction_prompt_v1 = """# Edit Refinement Task

Your task is to interpret and refine edits that have been automatically extracted from a source sentence and a corrected version. Given a sentence pair and a list of "Rough Edits", return a list of "Refined Edits". These should be sensible, human-interpretable revisions to the original text that follow the logic of "grammatical errors" and their corrections when possible.

## Input Format

source: The original sentence, which may have one or more errors.
corrected: A version of the source sentence with errors corrected.
rough_edits: A list of Rough Edit tuples of the format "(index, action, source_words, corrected_words)", which can take one of the following forms:
- (index, "replace", source_words, corrected_words)
- (index, "delete", source_words, "")
- (index, "insert", "", corrected_words)

## Output Format

The output is a JSON object, refined_edits, containing a list of Refined Edit dictionaries with the following keys: "index" (int), "action" (str), "source_words" (str), "corrected_words" (str), "attributed_edits" (List[int]). The attributed_edits are the indices of the Rough Edits that are the basis of this Refined Edit.

A Refined Edit dictionary can take one of the following forms:
- index, "replace", source_words, corrected_words, attributed_edits
- index, "delete", source_words, "", attributed_edits
- index, "insert", "", corrected_words, attributed_edits
- index, "relocate", source_words, corrected_words, attributed_edits

## Task Notes

- The number of Refined Edits does not have to match the number of Rough Edits. Rough Edits can be combined or split when forming Refined Edits, and the attributed_edits should reflect this.
- Every Rough Edit in the input must be handled, and should correspond to at least one Refined Edit.
- When writing a Refined Edit, all of the source_words and corrected_words must be taken from the set of source_words and corrected_words in the Rough Edits. Other words in the sentences can not be added.
- The "relocate" action primarily applies when the same words are deleted and then added elsewhere in the same order as a block. If a few words are reordered "within" the same phrase, it is the "replace" action.
- Only edits to adjacent words can be combined into a Refined Edit. Two related but distant changes must remain separate edits.

## Examples

### Example 1

Input:
source: I with my puppy go to teh store.
corrected: I go to teh store with my puppy.
rough_edits: [
    (0, "delete", "with my puppy", ""),
    (1, "insert", "", "with my puppy")
]

Output:
refined_edits: [
    {{"index": 0, "action": "relocate", "source_words": "with my puppy", "corrected_words": "with my puppy", "attributed_edits": [0, 1]}}
]

Notes: 
- A single Refined Edit with the "relocate" action is a sensible way to combine the two Rough Edits.
- There was an extra mistake that was not corrected ("teh"), which is ignored in this task because no Rough Edit describes it.

### Example 2

Input:
source: She don't see shoe you bought her.
corrected: She didn't see the shoes you bought her.
rough_edits: [
    (0, "replace", "do", "did"),
    (1, "insert", "", "the"),
    (2, "replace", "shoe", "shoes")]

Output:
refined_edits: [
    {{"index": 0, "action": "replace", "source_words": "do", "corrected_words": "did", "attributed_edits": [0]}},
    {{"index": 1, "action": "insert", "source_words": "", "corrected_words": "the", "attributed_edits": [1]}},
    {{"index": 2, "action": "replace", "source_words": "shoe", "corrected_words": "shoes", "attributed_edits": [2]}}
]

Notes: 
- The Rough Edits are fine as they are, pointing to three different issues.
- "do" and "did" are commonly tokenized separately from "n't", so it is okay to handle them at this level.

### Example 3

Input:
source: Despite of it is an industrial city. There is many shops and department stores.
corrected: Although it is an industrial city, there are many shops and department stores.
rough_edits: [
    (0, "replace", "Despite", "Although"),
    (1, "delete", "of", ""),
    (2, "replace", ". There", ", there"),
    (3, "replace", "is", "are")
]

Output:
refined_edits: [
    {{"index": 0, "action": "replace", "source_words": "Despite of", "corrected_words": "Although", "attributed_edits": [0, 1]}},
    {{"index": 1, "action": "replace", "source_words": ". There", "corrected_words": ", there", "attributed_edits": [2]}},
    {{"index": 2, "action": "replace", "source_words": "is", "corrected_words": "are", "attributed_edits": [3]}}
]

Notes: 
- The first two Rough Edits are combined, as it is more interpretable to consider "Although" as a replacement for "Despite of", which reads as a single lexical item.
- The other two Rough Edits point to two separate issues: punctuation and agreement.
- It would be wrong to merge "there" and "is" since only "is" needs to be changed for the agreement issue.

## Current Instance

Interpret and refine the Rough Edits in the following instance, following the format in the examples strictly to produce a refined_edits JSON object.

Input:
source: {source}
corrected: {corrected}
rough_edits: {rough_edits}
Output:
"""

feedback_prompt_v1 = """# Instructions

Your task is to analyze a sentence written by a learner of English and provide educational feedback for a single specific error at a time, using the target_edit as a guide.

You will define a highlight span for the error and associated words and provide feedback separated into two parts: an explanation of the error and a suggestion for correction.

## Input Format

The input will be one or more instances. An instance is a sentence and correction pair, along with a target_edit. Each instance will be in the following format:

- index: Integer starting from 0 used to differentiate between instances. Each output should correspond to a different instance.
- original_sentence: The original sentence, which may have one or more errors. The words that must be changed for the target_edit are marked with asterisks. An insertion is marked with "**".
- corrected_sentence A version of the original sentence with errors corrected. The words that must be changed for the target_edit are marked with asterisks. A deletion is marked with "**".
- target_edit: A change that must be made to the original sentence to fix one error in it. It corresponds to the words with asterisks in the original and corrected sentences.

## Output Format

Please return a JSON object named "feedback_list" containing a list of dictionaries, each with the following fields:

- index: Integer starting from 0 showing which input instance this feedback corresponds to.
- highlighted_sentence: A copy of the original sentence with a span "highlighted" by angle brackets.
- error_tag: A single tag that best describes the error underlying the target edit. This must be drawn from the following list: ['Part-of-Speech Confusion', 'Vocabulary Choice', 'Other Vocabulary Issue', 'Fixed Expressions', 'General Collocations', 'Phrasal Verbs', 'Verb + Preposition + Argument', 'Other Collocation Issue', 'Adverbs of Degree', 'Articles with Exceptional Words', 'Definite vs. Indefinite Article', 'Determiner-Noun Agreement', 'Indefinite Article Choice', 'Missing/Unnecessary Article', 'Other Determiner Error', 'Comparison: Comparative', 'Comparison: Equivalence', 'Comparison: Superlative', 'Causative', 'Conditional', 'Conjunction', 'Ditransitive Verbs', 'Dummy Subject', 'Expressions of Place', 'Expressions of Time', 'Go + ing', "Imperative/Let's", 'Infinitive', 'Modal', 'Negative Formation', 'Participle', 'Passive vs. Active Voice', 'Plural vs. Singular: Noun Countability', 'Plural vs. Singular: Noun Number', 'Possessive', 'Prepositions: Cause', 'Prepositions: Means/Agent', 'Prepositions: Status', 'Prepositions: Other', 'Pronoun Antecedent', 'Purpose Clause', 'Quantifier', 'Question Formation', 'Relative Clause', 'Result Clause', 'Subject-Verb Agreement', 'Tense: Continuous Aspect', 'Tense: Future Formation', 'Tense: Past Formation', 'Tense: Perfect', 'Tense: Tense Choice', 'That Clause', 'Verb Nominalization', 'Word Order', 'Other Grammar Issue', 'Capitalization', 'Colons', 'Commas', 'Contractions', 'Hyphenation', 'Parentheses', 'Quotation', 'Run-on Sentence', 'Semicolon', 'Spacing', 'Spelling', 'Terminal Punctuation', 'Other Punct/Mechanics Issue', 'Fragment: Incomplete Thought', 'Fragment: Missing Object', 'Fragment: Missing Subject', 'Fragment: Missing Verb', 'Grammatical Redundancy', 'Transition', 'Other Coherence/Cohesion Issue', 'Archaic or Formal Language', 'Casual or Informal Language', 'Potentially Rude/Insensitive', 'Stylistic Redundancy', 'Other Style/Register Issue']
- feedback_explanation: A brief, simple explanation of the error. This should tell the learner *what is wrong* and *why*. If a certain grammar point is involved, it can be mentioned.
- feedback_suggestion: A brief, simple comment outlining *what to do* to fix the error. This can be a hint or a direct correction, depending on the type of error in context. If the error is based on a broad rule of English, such as the verb forms used in different tenses, the suggestion should be a hint, not a direct correction. A hint should generally not provide an immediately usable replacement string. For lexical or idiom errors, a direct correction is fine. Example hint: 'Change "am" to a form that matches a third-person singular subject.' Example direct correction: 'Change "am" to "is."'

## Notes

- Both feedback_explanation and feedback_suggestion should be short and written in simple English. Certain grammatical terms can be used when necessary to explain an error or to make an appropriately indirect hint, but otherwise, simple is always better.
- The target_edit **must** be in highlighted_sentence's highlight, but the highlight can extend further than that. For example, in "He *am* my friend", only the word "am" is edited, but the highlighted_sentence should be "<He *am*> my friend" since these two words effectively show the subject-verb agreement error.
- If a word is missing from the original sentence and must be added (marked with "**"), the highlight **must** consist of the adjacent words to both the left and right of the insertion point.
  - Example 1 (inserting "is"): "<She ** my> sister."
  - Example 2 (inserting a comma): "I like <cats ** but> I'm allergic to them."
- Other than the added angle brackets, highlighted_sentence should be identical to the original_sentence, with no other corrections or alterations. It **must** have the complete text of the original_sentence, not just a part of it.
  - This applies even to whitespace such as line breaks or unstripped spaces. Copy them fathfully.

## Examples

### Example Input Instances

index: 0
original_sentence: I *is* a studunt.
corrected_sentence: I *am* a student.
target_edit: The replacement of "is" with "am"

index: 1
original_sentence: I is a *studunt*.
corrected_sentence: I am a *student*.
target_edit: The replacement of "studunt" with "student"

index: 2
original_sentence: I *with my puppy* went to teh store. 
corrected_sentence: I went to the store *with my puppy*. 
target_edit: The relocation of "with my puppy"

index: 3
original_sentence: She *do*n't see shoe you bought her.
corrected_sentence: She *did*n't see the shoes you bought her.
target_edit: The replacement of "do" with "did"

index: 4
original_sentence: She don't see ** shoe you bought her.
corrected_sentence: She didn't see *the* shoes you bought her.
target_edit: The insertion of "the"

index: 5
original_sentence: She don't see *shoe* you bought her.
corrected_sentence: She didn't see the *shoes* you bought her.
target_edit: The insertion of "the"

index: 6
original_sentence: When I came home ** my dog ran to meet me.  
corrected_sentence: When I came home *,* my dog ran to meet me.  
target_edit: The insertion of ","

### Example Outputs

```JSON
{{
    "feedback_list": [
        {{
            "index": 0,
            "highlighted_sentence": "<I *is*> a studunt.",
            "error_tag": "Subject-Verb Agreement",
            "feedback_explanation": "The verb \"to be\" changes depending on the subject. \"I\" can't go with \"is.\"",
            "feedback_suggestion": "Change \"is\" to a form that goes with \"I.\""
        }},
        {{
            "index": 1,
            "highlighted_sentence": "I is a <*studunt*>.",
            "error_tag": "Spelling",
            "feedback_explanation": "The word \"studunt\" seems to be misspelled.",
            "feedback_suggestion": "Did you mean \"student?\""
        }}
        {{
            "index": 2,
            "highlighted_sentence": "I <*with my puppy* went to teh store>. ",
            "error_tag": "Word Order",
            "feedback_explanation": "\"With my puppy\" is between the subject and verb, which is awkward. Prepositional phrases like this are usually after the verb and object.",
            "feedback_suggestion": "Move \"with my puppy\" after \"went to teh store\" to improve this sentence."
        }},
        {{
            "index": 3,
            "highlighted_sentence": "She <*do*n't see> shoe you bought her.",
            "error_tag": "Tense: Tense Choice",
            "feedback_explanation": "It seems this is describing a past event, so the verb should be in the past tense, not the present.",
            "feedback_suggestion": "Consider changing \"don't see\" to the past tense."
        }},
        {{
            "index": 4,
            "highlighted_sentence": "She don't <see ** shoe> you bought her.",
            "error_tag": "Missing/Unnecessary Article",
            "feedback_explanation": "This is about a specific pair of shoes, so it is necessary to use the definite article \"the.\"",
            "feedback_suggestion": "Add \"the\" before \"shoe.\""
        }},
        {{
            "index": 5,
            "highlighted_sentence": "She don't see <*shoe*> you bought her.",
            "error_tag": "Plural vs. Singular: Noun Number",
            "feedback_explanation": "This seems to be about more than one shoe, so the plural form is needed.",
            "feedback_suggestion": "Change \"shoe\" to be plural."
        }},
        {{
            "index": 6,
            "highlighted_sentence": "When I came <home ** my> dog ran to meet me.  ",
            "error_tag": "Commas",
            "feedback_explanation": "A comma is needed after introductory phrases to separate them from the main clause.",
            "feedback_suggestion": "Add a comma after \"home.\""
        }}
    ]
}}
```

## Current Task

Based on the above guidance and examples, please analyze the following instance(s) and provide highlights, error tags, and feedback.

### Input Instances

{input_batch}

### Outputs

"""