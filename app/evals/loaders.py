def load_logs(run_name=None, ids=None):
    from models.eval import QuestionLog
    from orm.sync_db import Session
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    with Session() as session:
        stmt = select(QuestionLog).options(selectinload(QuestionLog.question))
        if run_name:
            stmt = stmt.where(QuestionLog.run_name == run_name)
        if ids is not None:
            stmt = stmt.where(QuestionLog.id.in_(ids))
        return list(session.scalars(stmt))
