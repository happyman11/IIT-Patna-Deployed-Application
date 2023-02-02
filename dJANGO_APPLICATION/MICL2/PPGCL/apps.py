from django.apps import AppConfig


class PpgclConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "PPGCL"

    def ready(self):
        import PPGCL.Signals
