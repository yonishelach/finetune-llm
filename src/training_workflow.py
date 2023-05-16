import mlrun
from kfp import dsl


@dsl.pipeline(name="MLOps Bot Master Pipeline")
def kfpipeline(
    html_links: list[str],
    model_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    tokenizer_class: str,
    model_class: str,
    epochs: str,
    use_deepspeed: bool,
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Collect Dataset:
    collect_dataset_run = mlrun.run_function(
        function="data-collecting",
        handler="collect_html_to_text_files",
        params={"urls": html_links},
        outputs=["html-as-text-files"],
    )

    # Dataset Preparation:
    prepare_dataset_run = mlrun.run_function(
        function="data-preparing",
        handler="prepare_dataset",
        inputs={"source_dir": collect_dataset_run.outputs["html-as-text-files"]},
        outputs=["html-data"],
    )

    # Training:
    train_function = project.get_function("mpi-training")

    training_run = mlrun.run_function(
        function="mpi-training",
        name="train",
        inputs={"dataset": prepare_dataset_run.outputs["html-data"]},
        params={
            "model_name": model_name,
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "TRAIN_num_train_epochs": epochs,
            "TRAIN_fp16": True,
            "TRAIN_bf16": False,
            "TRAIN_per_device_train_batch_size": 4,
            "TRAIN_logging_strategy": "epoch",
            "use_deepspeed": use_deepspeed,
        },
        handler="train",
        outputs=["model"],
    )

    # evaluation:
    mlrun.run_function(
        function="training",
        name="evaluate",
        params={"model_path": training_run.outputs["model"]},
        inputs={"data": prepare_dataset_run.outputs["html-data"]},
        handler="evaluate",
    )

    # Create serving graph:
    serving_function = project.get_function("serving-mlopspedia")
    serving_function.with_limits(gpus=1)
    serving_function.spec.readiness_timeout = 3000

    # Set the topology and get the graph object:
    graph = serving_function.set_topology("flow", engine="async")
    # TODO: if not working add src.file
    graph.to(handler="preprocess", name="preprocess") \
        .to("LLMModelServer",
            name="mlopspedia",
            model_path=str(training_run.outputs["model"]),
            model_class="GPT2LMHeadModel",
            tokenizer_name="gpt2",
            tokenizer_class="GPT2Tokenizer",
            use_deepspeed=False) \
        .to(handler="postprocess", name="postprocess") \
        .to("ToxicityClassifierModelServer",
            name="toxicity-classifier",
            threshold=0.7).respond()

    # Deploy the serving function:
    deploy_return = mlrun.deploy_function("serving-mlopspedia")

    # Model server tester
    mlrun.run_function(
        function="testing",
        inputs={"dataset": prepare_dataset_run.outputs["html-data"]},
        params={
            "endpoint": deploy_return.outputs["endpoint"],
        },
    )
