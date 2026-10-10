"""Frozen TreeDebater.speak from 1f55a44; executed with ouragents globals.
Only the pre-incremental speaking method is overridden; tree preparation is shared.
"""
class BaselineTreeDebater(TreeDebater):
    def speak(self, prompt, max_time, time_control=False, history=None, **kwargs):
        call_id = next_call_id()
        ctx = dict(call_id=call_id, stage=self.status, side=self.side)
        set_speak_io_context(call_id, "tree_debater_speak")
        try:
            with timed_phase(logger, "tree_debater_speak", **ctx):
                self._add_message("user", prompt)
                if io_logging_enabled():
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Conversation-History",
                        body=json.dumps(self.conversation),
                        stage=self.status,
                        side=self.side,
                    )
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Prompt",
                        body=prompt,
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Conversation-History",
                        body=json.dumps(self.conversation),
                        stage=self.status,
                        side=self.side,
                    )
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Prompt",
                        body=prompt.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                logger.debug(
                    f"[timing-meta] call_id={call_id} speak_session=tree_debater_speak "
                    f"n_messages={len(self.conversation)}"
                )

                with timed_phase(logger, "main_get_response", **ctx):
                    response = self._get_response(self.conversation, **kwargs)
                if io_logging_enabled():
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Response-Before-Post-Process",
                        body=str(response).strip(),
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Response-Before-Post-Process",
                        body=str(response).strip(),
                        stage=self.status,
                        side=self.side,
                    )

                with timed_phase(logger, "revision_suggestion", pass_index=1, add_evidence=True, **ctx):
                    feedback_for_revision, new_evidence, allocation_plan, ori_statement = self._get_revision_suggestion(
                        statement=response, history=history, add_evidence=True, call_id=call_id, **kwargs
                    )
                with timed_phase(logger, "length_adjust", block=1, max_retry=1, **ctx):
                    response = self._length_adjust(
                        ori_statement,
                        feedback_for_revision,
                        new_evidence,
                        allocation_plan,
                        max_time,
                        max_retry=1,
                        call_id=call_id,
                        **kwargs,
                    )

                # Default to single-pass revision to reduce latency:
                # pass-1 revision + one length-adjust. Set single_pass_revision=False
                # (via config or kwargs) to restore the old two-pass behavior.
                single_pass_revision = kwargs.get(
                    "single_pass_revision",
                    getattr(self.config, "single_pass_revision", False),
                )
                if not single_pass_revision:
                    with timed_phase(logger, "revision_suggestion", pass_index=2, add_evidence=False, **ctx):
                        feedback_for_revision, new_evidence, _, _ = self._get_revision_suggestion(
                            statement=response, history=history, add_evidence=False, call_id=call_id, **kwargs
                        )

                    streaming_tts = kwargs.get("streaming_tts", getattr(self.config, "streaming_tts", False))
                    if not time_control or streaming_tts:
                        with timed_phase(logger, "length_adjust", block=2, max_retry=1, **ctx):
                            response = self._length_adjust(
                                response,
                                feedback_for_revision,
                                new_evidence,
                                allocation_plan,
                                max_time,
                                max_retry=1,
                                call_id=call_id,
                                **kwargs,
                            )
                    else:
                        with timed_phase(logger, "length_adjust", block=2, max_retry=10, **ctx):
                            response = self._length_adjust(
                                response,
                                feedback_for_revision,
                                new_evidence,
                                allocation_plan,
                                max_time,
                                max_retry=10,
                                call_id=call_id,
                                **kwargs,
                            )

                with timed_phase(logger, "post_process", **ctx):
                    out = super().post_process(response, max_time, time_control, **kwargs)
                return out
        finally:
            clear_speak_io_context()
