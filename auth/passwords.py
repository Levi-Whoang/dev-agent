"""密码哈希：只用标准库，不引入 bcrypt / passlib。

为什么不用 pbkdf2 存明文之外的东西：
- 明文密码一旦库泄露，等于把用户在其他网站的口令也交出去了；
- pbkdf2_hmac 是 Python 自带的，迭代 26 万次，单机暴力破解成本足够高；
- 每个用户独立盐值（salt），同样的密码在库里也是两个不同的哈希，
  避免“撞库 + 彩虹表”一次解开一片。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 260_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("密码不能为空")

    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)

    return "$".join(
        [
            _ALGORITHM,
            str(_ITERATIONS),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored.split("$")
        if algorithm != _ALGORITHM:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt, int(iterations)
        )
    except (ValueError, TypeError):
        return False

    # 用常量时间比较，避免根据耗时逐位猜哈希
    return hmac.compare_digest(actual, expected)
