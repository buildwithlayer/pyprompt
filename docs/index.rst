.. image:: _static/images/banner.png
   :width: 100%
   :alt: PyPrompt Banner
=================================

PyPrompt is a Python library for building and managing LLM prompts with token-aware pruning and priority management.

Contents
--------

.. toctree::
   :maxdepth: 2

   api

Features
--------
- Token-aware prompt management
- Priority-based content pruning
- Conversation history handling
- Template variable substitution
- Semantic text chunking
- OpenAI chat format support

Basic Usage
----------

.. code-block:: python

   from pyprompt import *

   # Create a prompt with priorities
   prompt = PromptElement(
       SystemMessage("You are an AI assistant.", priority=300),
       UserMessage("Input: {query}", priority=200)
   )

   # Render with token budget
   messages = render_prompt(
       prompt,
       encoding_func=encode,
       decoding_func=decode,
       props={"query": "Hello!"},
       total_budget=1000
   )

API Documentation
---------------

See the :doc:`api` section for detailed documentation.