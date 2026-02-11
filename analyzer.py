from authenticity_checker import check_image_authenticity
from predict import ml_predict
from fusion import final_verdict_fusion


def analyze_image(image_path):
    metadata_result, forensic_result = check_image_authenticity(image_path)
    ml_result = ml_predict(image_path)



    verdict, score = final_verdict_fusion(
        metadata_result,
        forensic_result,
        ml_result,
    )

    return {
        "metadata": metadata_result,
        "forensic": forensic_result,
        "ml": ml_result,
        "final_verdict": verdict,
        "confidence": score
    }
