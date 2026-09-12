from fastapi import HTTPException
from use_cases import card_wait


# every REST door that answers waits for the card here; a layout it cannot serve is a 409
def wait_for_the_card(*roles) -> None:
    try:
        card_wait.wait_for_the_card(*roles)
    except card_wait.CardHeld as e:
        raise HTTPException(
            status_code=503, detail=e.detail, headers={"Retry-After": str(e.retry_after)}
        ) from e
    except card_wait.CannotAnswer as e:
        raise HTTPException(status_code=409, detail=e.detail) from e
