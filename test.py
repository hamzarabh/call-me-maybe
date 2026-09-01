from lark import Lark, Token
import re
import numpy as np

json_grammar = """
    ?start: object
    object: LBRACE [pair (COMMA pair)*] RBRACE
    pair: STRING COLON value
    value: STRING | NUMBER | object
    
    LBRACE: "{"
    RBRACE: "}"
    COLON: ":"
    COMMA: ","
    STRING: "prompt" | "name" | "parameters"
    NUMBER: /[0-9]+/
    
    %ignore " "
    %ignore /\\n/
"""

parser = Lark(json_grammar, parser='lalr', lexer='contextual')
interactive_parser = parser.parse_interactive()

def get_allowed_regexes(interactive_parser):
    allowed_terminals = interactive_parser.accepts()
    regex_patterns = []
    
    for terminal in allowed_terminals:
        if terminal == 'LBRACE':
            regex_patterns.append(r"{")
        elif terminal == 'RBRACE':
            regex_patterns.append(r"}")
        elif terminal == 'COLON':
            regex_patterns.append(r":")
        elif terminal == 'COMMA':
            regex_patterns.append(r",")
        elif terminal == 'STRING':
            regex_patterns.append(r'"[^"]*"')
        elif terminal == 'NUMBER':
            regex_patterns.append(r'[0-9]+')

    return regex_patterns


rege = get_allowed_regexes(interactive_parser)

print(rege)
# print(interactive_parser.accepts())

for token in parser.lex("{prompt"):
    interactive_parser.feed_token(token)
rege = get_allowed_regexes(interactive_parser)

print(rege)
