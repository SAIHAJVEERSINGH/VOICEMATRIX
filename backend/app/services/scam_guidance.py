def get_scam_guidance(
    urgent_money_request: bool = False,
    asks_for_otp_or_pin: bool = False,
    caller_claims_emergency: bool = False,
) -> list[str]:
    advice = [
        "A voice detector alone cannot establish whether a call is a scam.",
        "Verify the caller using a separate, trusted communication channel.",
        "Never disclose banking PINs, passwords, or OTPs to a caller.",
    ]

    if urgent_money_request:
        advice.append(
            "Pause before transferring money and independently verify the request."
        )

    if asks_for_otp_or_pin:
        advice.append(
            "Do not share an OTP or PIN, even if the caller sounds familiar."
        )

    if caller_claims_emergency:
        advice.append(
            "Contact the person or organization using a number you already trust."
        )

    return advice
