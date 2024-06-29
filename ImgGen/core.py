"""
MIT License

Copyright (c) 2023-present japandotorg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import io
import json
from typing import Final, List, Optional
import aiohttp
import discord
from redbot.core import commands
from redbot.core.bot import Red
from redbot.core.utils.views import SetApiView


class DiffusionError(discord.errors.DiscordException):
    pass


class imgGen(commands.Cog):
    __author__: Final[List[str]] = ["tpn"]
    __version__: Final[str] = "0.1.1"

    BASE_URL: Final[str] = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    HEADERS: Final[dict] = {"Content-Type": "application/json"}

    def __init__(self, bot: Red) -> None:
        self.bot: Red = bot
        self.session: aiohttp.ClientSession = aiohttp.ClientSession()
        self.model_mapping = {
            "sdxl": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
            "dreamshaper": "@cf/lykon/dreamshaper-8-lcm",
            "sdxl-lightning": "@cf/bytedance/stable-diffusion-xl-lightning"
        }

    def format_help_for_context(self, ctx: commands.Context) -> str:
        pre_processed = super().format_help_for_context(ctx) or ""
        n = "\n" if "\n\n" not in pre_processed else ""
        text = [
            f"{pre_processed}{n}",
            f"Cog Version: **{self.__version__}**",
            f"Author: **{self.__author__}**",
        ]
        return "\n".join(text)

    async def cog_unload(self) -> None:
        if self.session:
            await self.session.close()

    async def _request(self, prompt: str, account_id: str, api_key: str, model: str) -> bytes:
        url = self.BASE_URL.format(account_id=account_id, model=model)
        headers = {**self.HEADERS, "Authorization": f"Bearer {api_key}"}
        payload = {"prompt": prompt}

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                raise DiffusionError(f"Error?: {response.status}")
            return await response.read()

    async def _generate_image(self, prompt: str, model: Optional[str] = None) -> bytes:
        token = await self.bot.get_shared_api_tokens("CFWorkersAI")
        account_id = token.get("account_id")
        api_key = token.get("api_key")
        default_model = token.get("model")
        if not account_id or not api_key or not default_model:
            raise DiffusionError("Setup not done. Use `set api CFWorkersAI account_id <your account id>`, `set api CFWorkersAI api_key <your api key>`, and `set api CFWorkersAI model <model>`.")

        model = model or default_model
        if model.lower() not in self.model_mapping:
            raise DiffusionError(f"`{model}` does not exist?")
        model = self.model_mapping.get(model.lower(), model)
        return await self._request(prompt, account_id, api_key, model)

    async def _image_to_file(self, image_data: bytes, prompt: str) -> discord.File:
        return discord.File(
            io.BytesIO(image_data),
            filename=f"{prompt.replace(' ', '_')}.png"
        )

    @commands.command(name="gen", aliases=["i"])
    async def _gen(self, ctx: commands.Context, *, prompt: str) -> None:
        await ctx.typing()
        prompt, *args = prompt.split(" ")
        model = None

        for arg in args:
            if arg.startswith("--model="):
                model = arg.split("=")[1]

        try:
            image_data = await self._generate_image(prompt, model)
        except DiffusionError as e:
            await ctx.send(
                f"Something went wrong...\n{e}",
                reference=ctx.message.to_reference(fail_if_not_exists=False),
                allowed_mentions=discord.AllowedMentions(replied_user=False),
            )
            return
        except aiohttp.ClientResponseError as e:
            await ctx.send(
                f"Error?: `{e.status}`\n{e.message}",
                reference=ctx.message.to_reference(fail_if_not_exists=False),
                allowed_mentions=discord.AllowedMentions(replied_user=False),
            )
            return
        file: discord.File = await self._image_to_file(image_data, prompt)
        await ctx.send(
            embed=discord.Embed(
                description=f"Prompt: {prompt}",
                color=await ctx.embed_color(),
            ),
            file=file,
        )