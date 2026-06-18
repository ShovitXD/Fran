import asyncio
import os
from dotenv import load_dotenv
from twitchAPI.twitch import Twitch
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.type import AuthScope
from twitchAPI.object.eventsub import (
    ChannelFollowEvent,
    ChannelSubscribeEvent,
    ChannelCheerEvent,
    ChannelPointsCustomRewardRedemptionAddEvent,
)


SECRETS_PATH = "D:/1- Importants/Notes/fran_secrets.env"
REDEMPTION_WHITELIST_PATH = "D:/Fran/twitch/fran_responses.txt"


def load_secrets():
    load_dotenv(SECRETS_PATH)

    return {
        "client_id": os.getenv("TWITCH_CLIENT_ID"),
        "client_secret": os.getenv("TWITCH_CLIENT_SECRET"),
        "oauth_token": os.getenv("TWITCH_OAUTH_TOKEN"),
        "refresh_token": os.getenv("TWITCH_REFRESH_TOKEN"),
        "channel": os.getenv("TWITCH_CHANNEL"),
    }


def load_whitelist():
    try:
        with open(REDEMPTION_WHITELIST_PATH, "r", encoding="utf-8") as f:
            entries = {line.strip().lower() for line in f if line.strip()}
        print(f"[Twitch] Loaded {len(entries)} whitelisted redemptions")
        return entries
    except FileNotFoundError:
        print(f"[Twitch] WARNING: Whitelist file not found at {REDEMPTION_WHITELIST_PATH}")
        return set()


# ─────────────────────────────────────────────
# EVENT HANDLERS
# ─────────────────────────────────────────────

async def on_follow(event: ChannelFollowEvent, input_queue):
    user = event.event.user_name
    print(f"[Twitch] Follow: {user}")
    input_queue.put(f"[Chat]: {user} just followed the stream!")


async def on_sub(event: ChannelSubscribeEvent, input_queue):
    user = event.event.user_name
    tier = event.event.tier
    tier_name = {"1000": "Tier 1", "2000": "Tier 2", "3000": "Tier 3"}.get(tier, "unknown tier")
    print(f"[Twitch] Sub: {user} ({tier_name})")
    input_queue.put(f"[Chat]: {user} just subscribed with {tier_name}!")


async def on_cheer(event: ChannelCheerEvent, input_queue):
    user = event.event.user_name or "Anonymous"
    bits = event.event.bits
    print(f"[Twitch] Cheer: {user} cheered {bits} bits")
    input_queue.put(f"[Chat]: {user} just cheered {bits} bits!")


async def on_redemption(event: ChannelPointsCustomRewardRedemptionAddEvent, input_queue, whitelist):
    user = event.event.user_name
    reward = event.event.reward.title
    user_input = event.event.user_input

    if reward.lower().strip() not in whitelist:
        print(f"[Twitch] Redemption ignored: {user} redeemed {reward}")
        return

    if user_input:
        print(f"[Twitch] Redemption: {user} redeemed {reward} — message: {user_input}")
        input_queue.put(f"[Chat]: {user} redeemed channel points: {reward} — they said: {user_input}")
    else:
        print(f"[Twitch] Redemption: {user} redeemed {reward}")
        input_queue.put(f"[Chat]: {user} redeemed channel points: {reward}!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def run_twitch_events(input_queue):
    secrets = load_secrets()
    whitelist = load_whitelist()

    twitch = await Twitch(secrets["client_id"], secrets["client_secret"])

    await twitch.set_user_authentication(
        secrets["oauth_token"],
        [
            AuthScope.BITS_READ,
            AuthScope.CHANNEL_READ_SUBSCRIPTIONS,
            AuthScope.CHANNEL_READ_REDEMPTIONS,
            AuthScope.MODERATOR_READ_FOLLOWERS,
        ],
        secrets["refresh_token"],
    )

    user_id = None
    async for user in twitch.get_users(logins=[secrets["channel"]]):
        user_id = user.id
        break

    print(f"[Twitch] Connected as {secrets['channel']} (ID: {user_id})")

    # ── async closures so the library gets proper async callbacks ──
    async def _on_follow(e): await on_follow(e, input_queue)
    async def _on_sub(e): await on_sub(e, input_queue)
    async def _on_cheer(e): await on_cheer(e, input_queue)
    async def _on_redemption(e): await on_redemption(e, input_queue, whitelist)

    eventsub = EventSubWebsocket(twitch)
    eventsub.start()

    await eventsub.listen_channel_follow_v2(user_id, user_id, _on_follow)
    await eventsub.listen_channel_subscribe(user_id, _on_sub)
    await eventsub.listen_channel_cheer(user_id, _on_cheer)
    await eventsub.listen_channel_points_custom_reward_redemption_add(user_id, _on_redemption)

    print("[Twitch] Listening for events...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await eventsub.stop()
        await twitch.close()