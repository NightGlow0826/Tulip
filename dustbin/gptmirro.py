#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : gptmirro.py
@Author  : Gan Yuyang
@Time    : 2023/4/17 13:44
"""
import time

import fake_useragent
import requests

session_url = 'https://chatbot.theb.ai/'
session = requests.Session()
init_status = session.get(session_url, headers={
    'User-Agent': fake_useragent.UserAgent().random,
})

url = f'https://chatbot.theb.ai/api/chat-process'
params = {
    "authority": "chatbot.theb.ai",
    "method": "POST",
    "path": "/api/chat-process",
    "scheme": "https",
    "accept": "application/json, text/plain, */*",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "content-length": "117",
    "content-type": "application/json",
    "cookie": "__cf_bm=Q5KQkU6Yx3QdU9rLsNcu4knF33Gyy7Ir5TyRWvXEQ18-1681710190-0-Aab0jPab7ar/RYUXhO42aZRXqZV6nFcViTDr2c2mMwWihSeS10R5sMWVL4Y/mRYG6z3UiQnvtKXpev5J9oapZa3ywED5IoXmxbMb0lvskfMm",
    "origin": "https://chatbot.theb.ai",
    "referer": "https://chatbot.theb.ai/",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.39",
}

# data = {'prompt': "没有看到呢",
#         'options': {'parentMessageId': "chatcmpl-76H0pxoi74GKEvqjz7TauRzwpdNfS"}}
#
# resp = session.post(f'https://chatbot.theb.ai/api/chat-process', params=params, data=data)
# resp.encoding = 'utf-8'
# print(resp.url)
# print(resp.status_code)
# print(resp.text)
print(time.time())
