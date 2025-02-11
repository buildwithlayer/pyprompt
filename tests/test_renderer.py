from conftest import decode, encode, generic_prompt

from pyprompt import *


def test_render_prompt(generic_prompt: PromptElement):
    prompt = PromptElement(
        AssistantMessage(
            "You are a SQL expert...",
            priority=300,
        ),
        UserMessage(
            "Here are the creation scripts that were used to create the tables in my database...",
            priority=200,
        ),
        UserMessage(
            TextChunk("{user_query}", break_on=" "),
            priority=100,
        ),
    )
    user_query = "CREATE TABLE Persons (\n\tPersonID int,\n\tLastName varchar(255),\n\tFirstName varchar(255),\n\tAddress varchar(255),\n\tCity varchar(255)\n);"
    props = {"user_query": user_query}

    assert render_prompt(prompt, encode, decode, props, 67) == [
        {"role": "assistant", "content": "You are a SQL expert..."},
        {
            "role": "user",
            "content": "Here are the creation scripts that were used to create the tables in my database...",
        },
        {"role": "user", "content": user_query},
    ]

    assert render_prompt(prompt, encode, decode, props, 66) == [
        {"role": "assistant", "content": "You are a SQL expert..."},
        {
            "role": "user",
            "content": "Here are the creation scripts that were used to create the tables in my database...",
        },
        {
            "role": "user",
            "content": "CREATE TABLE Persons (\n\tPersonID int,\n\tLastName varchar(255),\n\tFirstName varchar(255),\n\tAddress varchar(255),\n\tCity",
        },
    ]

    assert render_prompt(prompt, encode, decode, props, 60) == [
        {"role": "assistant", "content": "You are a SQL expert..."},
        {
            "role": "user",
            "content": "Here are the creation scripts that were used to create the tables in my database...",
        },
        {
            "role": "user",
            "content": "CREATE TABLE Persons (\n\tPersonID int,\n\tLastName varchar(255),\n\tFirstName varchar(255),\n\tAddress",
        },
    ]

    assert render_prompt(generic_prompt, encode, decode, dict(), 43) == [
        {
            "role": "system",
            "content": "You are a helpful assistant that cheers people up.",
        },
        {
            "role": "system",
            "content": "How are you?",
        },
        {
            "role": "system",
            "content": "I am fantastic. How are you?",
        },
        {"role": "user", "content": "What time is it?"},
    ]
