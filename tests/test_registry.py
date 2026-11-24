def test_sppe_type_is_resolved_by_name():
    from hybrik.models import builder

    # Simulates building straight from a config: ``cfg.MODEL.TYPE`` is a string
    # and nothing has imported the model class yet.
    builder._register_module_from_name('Simple3DPoseBaseSMPL', builder.SPPE)
    assert builder.SPPE.get('Simple3DPoseBaseSMPL') is not None


def test_unknown_type_is_left_to_the_registry():
    from hybrik.models import builder

    builder._register_module_from_name('DoesNotExist', builder.SPPE)
    assert builder.SPPE.get('DoesNotExist') is None
