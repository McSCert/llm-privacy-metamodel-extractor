import json
import os
from privacy_schema.models import (
    JurisdictionModel,
    RegulationModel,
    ActorModel,
    PersonalDataModel,
    ProcessingActivityModel,
    DataTransferModel,
    PurposeModel,
    LegalBasisModel,
    ConstraintModel,
    RightModel,
    RetentionPolicyModel,
    ConsentWithdrawalModel,
    PolicyStatementModel,
    PrivacyPolicyModel,
)

def generate_and_save_schemas(output_dir: str = "schemas/"):
    """
    Compiles Pydantic models from privacy_schema/models.py
    and saves each class schema as a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = {
        "JurisdictionModel": JurisdictionModel,
        "RegulationModel": RegulationModel,
        "ActorModel": ActorModel,
        "PersonalDataModel": PersonalDataModel,
        "ProcessingActivityModel": ProcessingActivityModel,
        "DataTransferModel": DataTransferModel,
        "PurposeModel": PurposeModel,
        "LegalBasisModel": LegalBasisModel,
        "ConstraintModel": ConstraintModel,
        "RightModel": RightModel,
        "RetentionPolicyModel": RetentionPolicyModel,
        "ConsentWithdrawalModel": ConsentWithdrawalModel,
        "PolicyStatementModel": PolicyStatementModel,
        "PrivacyPolicyModel": PrivacyPolicyModel,
    }

    # Save individual schema per class
    for class_name, model_class in classes.items():
        schema = model_class.model_json_schema()
        output_path = os.path.join(output_dir, f"{class_name}.json")
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Saved: {output_path}")

    # Save combined schema
    combined = {
        name: cls.model_json_schema()
        for name, cls in classes.items()
    }
    combined_path = os.path.join(output_dir, "all_schemas.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined schema: {combined_path}")

if __name__ == "__main__":
    generate_and_save_schemas()