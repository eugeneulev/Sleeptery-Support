import litellm


async def generate_response(
    model: str, system_prompt: str, user_message: str
) -> str:
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
