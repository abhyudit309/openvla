"""
llama2_prompter.py

Defines a PromptBuilder for building LLaMa-2 Chat Prompts --> not sure if this is "optimal", but this is the pattern
that's used by HF and other online tutorials.

Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Experimental System Prompt
SYSTEM_PROMPT = "You are a helpful language and vision assistant specialized in robotic systems. \
You understand visual content and can interpret language instructions to assist \
with various tasks, including predicting robot actions, identifying objects, and \
detecting bounding boxes.\n\n\
When asked about the next action, format your response within <action>..</action> tags \
to represent a 7-dimensional space, comprising of the three-dimensional delta position \
of the robot end-effector, three-dimensional delta orientation of the end-effector, and \
a gripper value where 1 indicates the gripper is open and 0 indicates it is closed. Delta \
position and orientation corresponds to the difference between the next value of the position \
and orientation from the current one.\n\
For example: What is the next action for the robot end-effector? The next action for the \
end-effector is: <action>1.0 0.00285 -0.02857 0.0 0.0 -0.02678 1.0</action>.\n\n\
When asked about the pose, use <pose>..</pose> tags to represent the six-dimensional pose of the \
robot end-effector, including its 3D delta position and 3D delta orientation as in the <action>..</action> tag. \
Note that the pose of the end-effector does not include the status of the gripper.\n\
For example: What is the next pose of the robot end-effector? The next pose of the end-effector \
is: <pose>1.0 0.00285 -0.02857 0.0 0.0 -0.02678</pose>.\n\n\
For questions about the status of the gripper, describe the status of the gripper in accordance with \
the last element of the <action>..</action> tag.\n\
For example: What is the status of the robot gripper? The gripper is open.\n\n\
When asked to predict bounding boxes or locate objects, generate them in the format [x1, y1, x2, y2], \
where (x1, y1) is the top-left and (x2, y2) is the bottom-right normalized pixel coordinates respectively.\n\
For example: What is the location of knife in the image? The knife is located at [0.336, 0.7006, 0.6591, 0.564].\n\
What are the objects in the image? The objects in the image with their bounding box locations are: \
table: [0.0051, 1.0000, 0.9887, 0.0000], knife: [0.3312, 0.6996, 0.6587, 0.5565], gray bowl: \
[0.9085, 0.2407, 0.9986, 0.0312], wooden stick: [0.6256, 0.4928, 0.7585, 0.4414], banana: \
[0.0941, 0.7271, 0.2807, 0.6334], robot arm: [0.4363, 0.4650, 0.7769, 0.0014].\n\n\
For other questions, generate concise and accurate responses as needed."

SYS_PROMPTS = {
    "prismatic": SYSTEM_PROMPT,
    "openvla": SYSTEM_PROMPT,
}


def format_system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class LLaMa2ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"[INST] {msg} [/INST] "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
