from real_db import pytestmark  # noqa: F401
from sqlalchemy import text


# a prompt version an answer log recorded is refused deletion: a later POST would give its number to another text
def test_a_prompt_version_runs_recorded_is_not_deleted(db):
    import asyncio

    from api.v1 import prompt as route
    from fastapi import HTTPException
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    with db.connect() as c:
        c.execute(text("TRUNCATE prompts, question_logs CASCADE"))
        c.execute(text("INSERT INTO prompts (id, purpose, version, template, active) VALUES"
                       " (1, 'judge.faithfulness', 7, 'old', false), (2, 'judge.faithfulness', 8, 'unused', false)"))
        c.execute(text("INSERT INTO question_logs (answered, prompts) VALUES (true, '{\"judge_faithfulness\": 7}')"))
        c.commit()
    async def both():
        engine = create_async_engine(db.url.set(drivername="postgresql+asyncpg"))
        try:
            async with AsyncSession(engine) as session:
                try:
                    await route.delete_prompt(1, session)
                    refused = None
                except HTTPException as e:
                    refused = e.status_code
            async with AsyncSession(engine) as session:
                kept = await route.delete_prompt(2, session)
            return refused, kept.version
        finally:
            await engine.dispose()

    assert asyncio.run(both()) == (409, 8)
