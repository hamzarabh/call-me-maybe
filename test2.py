import json

#  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
t = """{
  "name": "fn_substitute_string_with_regex",
  "prompt": "Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "([0-9]+)",
    "replacement": "NUMBERS"
  }
}"""
a = json.loads(t)

print(a)
