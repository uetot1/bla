from dcvc_rt.src.models.image_model import DMCI


LAMBDA_MIN = 1.0
LAMBDA_MAX = 64.0
LAMBDA_MAPPING = "geometric_qp_1_64"


def lambda_for_qp(qp):
    """Paper Eq. 5: lambda(q) = 64^(q/63)."""
    last_qp = DMCI.get_qp_num() - 1
    if not 0 <= qp <= last_qp:
        raise ValueError(f"base QP must be in the range 0..{last_qp}")
    return LAMBDA_MIN * (LAMBDA_MAX / LAMBDA_MIN) ** (qp / last_qp)
