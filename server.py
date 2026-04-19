import logging
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCIceCandidate,
)
from LyingVideoTrack import LyingVideoTrack
from contextlib import asynccontextmanager
from fastapi.templating import Jinja2Templates
from typing import Dict, Optional
from aiortc.contrib.media import MediaRelay


logging.basicConfig(level=logging.INFO)

templates = Jinja2Templates(directory="templates")

pcs: Dict[str, RTCPeerConnection] = {}
relay = MediaRelay()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    # await initialize_database()
    # await initialize_scheduler()
    print("All startup tasks completed.")
    yield
    # Shutdown tasks
    # await clean_up_database()
    # await clean_up_scheduler()
    await on_shutdown()
    print("All shutdown tasks completed.")


async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    if coros:
        await __import__("asyncio").gather(*coros)
    pcs.clear()


class OfferBody(BaseModel):
    sdp: str
    type: str


class CandidateData(BaseModel):
    candidate: str
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None


class CandidateBody(BaseModel):
    sessionId: str
    candidate: Optional[CandidateData] = None


class CloseBody(BaseModel):
    sessionId: str


def make_pc() -> RTCPeerConnection:
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(
                urls=[
                    "turn:1.234.5.55:3478?transport=udp",
                    "turn:1.234.5.55:3478?transport=tcp",
                ],
                username="black",
                credential="323285",
            ),
        ]
    )
    return RTCPeerConnection(configuration=config)


async def cleanup_pc(session_id: str):
    pc = pcs.pop(session_id, None)
    if pc:
        await pc.close()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


@app.post("/offer")
async def offer(body: OfferBody):

    pc = make_pc()
    session_id = str(uuid.uuid4())
    pcs[session_id] = pc

    pc_id = f"PeerConnection({id(pc)})"
    logging.info("%s created", pc_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logging.info("%s ICE state: %s", pc_id, pc.iceConnectionState)
        if pc.iceConnectionState in ("failed", "closed", "disconnected"):
            await cleanup_pc(session_id)
            logging.info("%s closed", pc_id)

    @pc.on("track")
    def on_track(track):
        logging.info("%s Track received: kind=%s", pc_id, track.kind)

        if track.kind == "video":
            # 들어오는 원본 비디오를 기반으로
            # 가공 비디오 트랙 생성 후 다시 송출

            # annotated_track = AnnotatedVideoTrack(track)
            annotated_track = LyingVideoTrack(track)
            pc.addTrack(annotated_track)
            logging.info("%s annotated track added", pc_id)

        @track.on("ended")
        async def on_ended():
            logging.info("%s Track ended: kind=%s", pc_id, track.kind)

    offer = RTCSessionDescription(sdp=body.sdp, type=body.type)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sessionId": session_id,
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


@app.post("/candidate")
async def candidate(body: CandidateBody):
    pc = pcs.get(body.sessionId)
    if not pc:
        return {"ok": False, "error": "session not found"}

    try:
        if body.candidate is None:
            # end-of-candidates
            await pc.addIceCandidate(None)
        else:
            c = body.candidate

            # 브라우저에서 받은 candidate 문자열을 aiortc용 객체로 변환
            ice = candidate_from_sdp(
                sdp=c.candidate,
                sdpMid=c.sdpMid,
                sdpMLineIndex=c.sdpMLineIndex,
            )
            await pc.addIceCandidate(ice)

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/close")
async def close(body: CloseBody):
    await cleanup_pc(body.sessionId)
    return {"ok": True}


def candidate_from_sdp(
    sdp: str, sdpMid: Optional[str], sdpMLineIndex: Optional[int]
) -> RTCIceCandidate:
    """
    예:
    candidate:842163049 1 udp 1677729535 192.168.0.10 61764 typ srflx raddr 0.0.0.0 rport 0
    """
    parts = sdp.split()

    # "candidate:foundation"
    foundation = parts[0].split(":")[1]
    component = int(parts[1])
    protocol = parts[2].lower()
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])

    cand_type = None
    relatedAddress = None
    relatedPort = None
    tcpType = None

    i = 6
    while i < len(parts):
        token = parts[i]

        if token == "typ":
            cand_type = parts[i + 1]
            i += 2
        elif token == "raddr":
            relatedAddress = parts[i + 1]
            i += 2
        elif token == "rport":
            relatedPort = int(parts[i + 1])
            i += 2
        elif token == "tcptype":
            tcpType = parts[i + 1]
            i += 2
        else:
            i += 1

    return RTCIceCandidate(
        component=component,
        foundation=foundation,
        ip=ip,
        port=port,
        priority=priority,
        protocol=protocol,
        type=cand_type,
        relatedAddress=relatedAddress,
        relatedPort=relatedPort,
        sdpMid=sdpMid,
        sdpMLineIndex=sdpMLineIndex,
        tcpType=tcpType,
    )
