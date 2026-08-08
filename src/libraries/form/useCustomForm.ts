import { useForm, UseFormProps } from 'react-hook-form';

/**
 * A custom wrapper around react-hook-form to standardize form configuration
 * across the application.
 */
export function useCustomForm<TFieldValues extends Record<string, any>>(
  options?: UseFormProps<TFieldValues>
) {
  const form = useForm<TFieldValues>({
    mode: 'onBlur',
    ...options,
  });

  return form;
}
