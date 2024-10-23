from abc import ABC
from typing import List

from pyasp import Command, logger
from pyasp.asp import AspStepBase


class AmesStereoPipelineError(Exception):
    """Custom exception for Ames Stereo Pipeline errors."""

    pass


class AmesStereoPipelineBase(ABC):
    _sensor = ""
    _pipeline = []

    def __init__(self, steps: List[AspStepBase] | dict[str, AspStepBase]):
        if isinstance(steps, dict):
            steps = [step for step in steps.values()]

        # Check if all steps are AspStepBase objects
        for step in steps:
            if not isinstance(step, (AspStepBase, Command)):
                raise TypeError(
                    f"Invalid {step} in steps. All steps must be AspStepBase or Command objects."
                )
        self._pipeline = steps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} Pipeline with steps: {self._pipeline}"

    @property
    def steps(self) -> List[AspStepBase]:
        return self._pipeline

    def run(self):
        """Run the entire stereo pipeline."""
        logger.info("Starting the stereo pipeline...\n")
        for step in self._pipeline:
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(
                f"Finished step: {step}. Time elapsed: {step.elapsed_time:.2f}s"
            )
            logger.info("---------------------------------------------\n")

    def resume_from_step(self, step_number: int):
        """Resume the stereo pipeline from a specific step."""
        if step_number >= len(self._pipeline):
            raise AmesStereoPipelineError(
                f"Invalid step number {step_number}. Must be less than {len(self._pipeline)}."
            )

        logger.info(f"Resuming the stereo pipeline from step {step_number}...\n")
        for i, step in enumerate(self._pipeline):
            if i < step_number:
                logger.info(f"Skipping step: {step}")
                continue
            logger.info(f"Running step: {step}")
            step.run()
            logger.info(
                f"Finished step: {step}. Time elapsed: {step.elapsed_time:.2f}s"
            )
            logger.info("---------------------------------------------\n")
        logger.info("Finished running the stereo pipeline.\n")
